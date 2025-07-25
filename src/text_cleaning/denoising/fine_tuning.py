import json
import logging
from pathlib import Path
from strenum import StrEnum
from typing import Literal
from datetime import datetime

from huggingface_hub import HfApi

from text_cleaning.constants import DATA_DIR, BASE_DIR, SYNTHETIC_OCR_DATASET_PATH, SYNTHETIC_CLEAN_DATASET_PATH
from text_cleaning.denoising.denoising import MAX_CONTEXT_TOKENS
from text_cleaning.utils import load_data, save_data, split_dataset, cache_model_and_tokenizer, Model

logger = logging.getLogger(__name__)

HF_DATASET_REPO_BASE_NAME = "ocr_denoising"
HF_USER_ID = "ClemensK"

SFT_DIR = DATA_DIR / "fine_tuning"
SFT_DATASET_DIR = SFT_DIR / "datasets"
SFT_TRAIN_CONFIG_DIR = SFT_DIR / "train_configs"
SFT_MODEL_DIR = SFT_DIR / "models"


class FineTuningDataset(StrEnum):
    THE_VAMPYRE = f"{HF_DATASET_REPO_BASE_NAME}-the_vampyre"
    SYNTHETIC = f"{HF_DATASET_REPO_BASE_NAME}-synthetic"


class LLaMAFactoryConfigs:
    def __init__(self, model: Model = Model.GEMMA, dataset: FineTuningDataset = FineTuningDataset.THE_VAMPYRE):
        self.model = model
        self.dataset = dataset
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = f"{model.name.lower()}_{dataset.name.lower()}_{timestamp}"
        self.train_config_path = SFT_TRAIN_CONFIG_DIR / f"ocr-{model.name.lower()}-{dataset.name.lower()}-config.json"
        self.export_config_path = SFT_MODEL_DIR / f"merged-{model.name.lower()}-{dataset.name.lower()}-config.json"
        self.sft_output_dir_name = ".." / (
            SFT_MODEL_DIR / f"sft_{self.model.name.lower()}_{self.dataset.name.lower()}"
        ).relative_to(BASE_DIR)
        self.sft_export_dir_name = ".." / (
            SFT_MODEL_DIR / f"export_{self.model.name.lower()}_{self.dataset.name.lower()}"
        ).relative_to(BASE_DIR)

        self.train_config_args = self._get_train_config_args()
        self.export_config_args = self._get_export_config_args()

    def generate_train_config(self):
        SFT_TRAIN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.train_config_path, "w", encoding="utf-8") as f:
            json.dump(self.train_config_args, f, indent=2)

    def generate_export_config(self):
        SFT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.export_config_path, "w", encoding="utf-8") as f:
            json.dump(self.export_config_args, f, indent=2)

    def _get_export_config_args(self):
        if self.model == Model.GEMMA:
            template = "gemma"
            model_id = "gemma-3-1b-it-ocr-denoising-en"
            finetuning_type = "lora"
        elif self.model == Model.LLAMA:
            template = "llama3"
            model_id = "Llama-3.2-1B-Instruct-ocr-denoising-en"
            finetuning_type = "lora"
        elif self.model == Model.MINERVA:
            template = "llama3"
            model_id = "Minerva-1B-base-v1.0-ocr-denoising-en"
            finetuning_type = "full"
        else:
            raise ValueError(f"Model {self.model} not supported")
        return dict(
            model_name_or_path=self.model.value,
            adapter_name_or_path=str(self.sft_output_dir_name),  # load the saved LoRA adapters
            template=template,  # same to the one in training
            finetuning_type=finetuning_type,  # same to the one in training
            export_dir=str(self.sft_export_dir_name),  # path to save the merged model
            export_size=2,  # the file shard size (in GB) of the merged model
            export_device="cpu",  # the device used in export, can be chosen from `cpu` and `cuda`
            export_hub_model_id=model_id,  # your Hugging Face hub model ID
        )

    def _get_train_config_args(self):
        if self.model == Model.GEMMA:
            return self._get_gemma_train_config_args()
        elif self.model == Model.LLAMA:
            return self._get_llama_train_config_args()
        elif self.model == Model.MINERVA:
            return self._get_minerva_train_config_args()
        else:
            raise ValueError(f"Model {self.model} not supported")

    def _get_gemma_train_config_args(self):
        return dict(
            # model
            model_name_or_path=self.model.value,
            # method
            stage="sft",  # do supervised fine-tuning
            do_train=True,
            finetuning_type="lora",  # use LoRA adapters to save memory
            lora_target="all",  # attach LoRA adapters to all linear layers
            # dataset
            dataset=str(self.dataset),  # use the custom dataset
            template="gemma",  # use Gemma prompt template
            max_samples=560000,
            # output
            output_dir=str(self.sft_output_dir_name),  # the path to save LoRA adapters
            logging_steps=10,  # log every 10 steps
            save_steps=1000,  # save checkpoint every 1000 steps
            overwrite_output_dir=True,
            # train
            per_device_train_batch_size=2,  # the batch size
            gradient_accumulation_steps=4,  # the gradient accumulation steps
            learning_rate=5e-5,  # the learning rate
            num_train_epochs=3.0,  # the epochs of training
            lr_scheduler_type="cosine",  # use cosine learning rate scheduler
            warmup_ratio=0.1,  # use warmup scheduler
            max_grad_norm=1.0,  # clip gradient norm to 1.0
            quantization_bit=4,  # use 4-bit QLoRA (gemma seems to require it)
            loraplus_lr_ratio=16.0,  # use LoRA+ algorithm with lambda=16.0
            fp16=True,  # use float16 mixed precision training
            # logging
            report_to="wandb",
            run_name=self.run_name,
        )

    def _get_llama_train_config_args(self):
        return dict(
            # model
            model_name_or_path=self.model.value,
            # method
            stage="sft",  # do supervised fine-tuning
            do_train=True,
            finetuning_type="lora",  # use LoRA adapters to save memory
            lora_target="all",  # attach LoRA adapters to all linear layers
            # dataset
            dataset=self.dataset.value,  # use custom dataset
            template="llama3",  # use llama3 prompt template
            cutoff_len=MAX_CONTEXT_TOKENS,
            max_samples=560000,
            overwrite_cache=True,
            preprocessing_num_workers=16,
            # output
            output_dir=str(self.sft_output_dir_name),
            logging_steps=10,
            save_steps=500,
            plot_loss=True,
            overwrite_output_dir=True,
            # train
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            num_train_epochs=3.0,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_grad_norm=1.0,  # clip gradient norm to 1.0
            loraplus_lr_ratio=16.0,  # use LoRA+
            fp16=True,  # use float16 mixed precision training
            ddp_timeout=180000000,
            # eval
            val_size=0.1,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=100,
            # logging
            report_to="wandb",
            run_name=self.run_name,
        )

    def _get_minerva_train_config_args(self):
        return dict(
            # model
            model_name_or_path=self.model.value,
            # method
            stage="sft",  # do supervised fine-tuning
            do_train=True,
            finetuning_type="full",
            use_badam=True,
            badam_mode="layer",
            badam_switch_mode="ascending",
            badam_switch_interval=50,
            badam_verbose=2,
            deepspeed="examples/deepspeed/ds_z3_config.json",
            # dataset
            dataset=self.dataset.value,  # use custom dataset
            template="llama3",  # use llama3 prompt template
            cutoff_len=MAX_CONTEXT_TOKENS,
            max_samples=560000,
            overwrite_cache=True,
            preprocessing_num_workers=16,
            # output
            output_dir=str(self.sft_output_dir_name),
            logging_steps=10,
            save_steps=500,
            plot_loss=True,
            overwrite_output_dir=True,
            # train
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            num_train_epochs=3.0,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            ddp_timeout=180000000,
            # eval
            val_size=0.01,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=100,
            # logging
            report_to="wandb",
            run_name=self.run_name,
        )


def _prepare_fine_tuning_task(
    configs: LLaMAFactoryConfigs,
) -> None:
    """
    Generate the LLaMA-Factory training and merging config files for a given model.
    """
    configs.generate_train_config()
    configs.generate_export_config()


def get_repo_url(dataset: FineTuningDataset) -> str:
    return f"{HF_USER_ID}/{dataset.value}"


def _prepare_ocr_fine_tuning_dataset(
    dataset: FineTuningDataset,
    ocr_file: str | Path,
    clean_file: str | Path,
    train_out_dir: str | Path = SFT_DATASET_DIR / "llama-factory",
    test_out_dir: str | Path = SFT_DATASET_DIR / "ocr",
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Prepare a LLaMA-Factory dataset from given OCR data for fine-tuning an OCR model.

    It uses the noisy OCR dataset and the cleaned ground truth dataset to generate a fine-tuning dataset
    in the format needed for LLaMA-Factory.

    The format of the fine-tuning dataset is:
    {
        "instruction": "Clean the following OCR text:",
        "input": ocr_text,
        "output": clean_text
    }

    Args:
        noisy_file: Path to the noisy OCR dataset file (JSON).
        clean_file: Path to the cleaned ground truth dataset file (JSON).
        train_out_dir: Path to save the generated fine-tuning dataset for training via LLaMA-Factory.
        test_out_dir: Path to save the noisy test dataset in original ocr format for later usage.
        test_ratio: Ratio of testing set size to total dataset size.
        seed: Random seed for reproducibility.
    """
    train_out_dir = Path(train_out_dir)
    test_out_dir = Path(test_out_dir)
    ocr_file = Path(ocr_file)
    clean_file = Path(clean_file)

    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    train_file = train_out_dir / f"{dataset.value}.json"

    noisy_data = load_data(ocr_file)
    clean_data = load_data(clean_file)

    # split pages of given dataset into train and test
    train_noisy_data, test_noisy_data = split_dataset(noisy_data, test_ratio=test_ratio, seed=seed)

    # create a dataset for each page number
    train_noisy_dataset = []
    for page_number in train_noisy_data.keys():
        train_noisy_dataset.append(
            {
                "instruction": "Clean the following OCR text:",
                "input": train_noisy_data[page_number],
                "output": clean_data[page_number],
            }
        )

    # save the LLaMA-Factory train dataset
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_noisy_dataset, f, indent=2)

    # save noisy test data in original ocr format for later usage
    save_data(test_out_dir / f"test-{ocr_file.name}", test_noisy_data)

    # push to huggingface
    try:
        repo_url = get_repo_url(dataset)
        api = HfApi()

        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_url, repo_type="dataset", exist_ok=True)
        except Exception as e:
            logger.warning(f"Repository creation failed (might already exist): {e}")

        # Upload the JSON file
        filename = train_file.name
        api.upload_file(
            path_or_fileobj=train_file,
            path_in_repo=filename,
            repo_id=repo_url,
            repo_type="dataset",
            commit_message="Update dataset file",
        )
        logger.info(f"Dataset pushed to Hugging Face Hub: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to push dataset to Hugging Face Hub: {e}")


def get_model_from_str(value: str) -> Model:
    if value == "gemma":
        return Model.GEMMA
    elif value == "llama":
        return Model.LLAMA
    elif value == "minerva":
        return Model.MINERVA
    else:
        raise ValueError(f"Model {value} not supported")


def get_dataset_from_str(value: str) -> FineTuningDataset:
    if value == "the_vampyre":
        return FineTuningDataset.THE_VAMPYRE
    elif value == "synthetic":
        return FineTuningDataset.SYNTHETIC
    else:
        raise ValueError(f"Dataset {value} not supported")


def prepare_fine_tuning(
    models: tuple[Literal["gemma", "llama", "minerva"], ...] = ("gemma", "llama", "minerva"),
    datasets: tuple[Literal["the_vampyre", "synthetic"], ...] = ("the_vampyre", "synthetic"),
    generate_files: bool = False,
    cache_models: bool = False,
) -> None:
    """Prepare the fine-tuning dataset and generate the LLaMA-Factory config file."""
    models_obj = [get_model_from_str(model) for model in models]
    datasets_obj = [get_dataset_from_str(dataset) for dataset in datasets]

    if cache_models:
        for model in models_obj:
            cache_model_and_tokenizer(model)

    for dataset in datasets_obj:
        if generate_files:
            for model in models_obj:
                configs = LLaMAFactoryConfigs(model=model, dataset=dataset)
                _prepare_fine_tuning_task(configs)

            if dataset == FineTuningDataset.THE_VAMPYRE:
                ocr_file = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_ocr.json"
                clean_file = DATA_DIR / "ocr_datasets" / "eng" / "the_vampyre_clean.json"
                test_ratio = 0.5
            elif dataset == FineTuningDataset.SYNTHETIC:
                ocr_file = SYNTHETIC_OCR_DATASET_PATH
                clean_file = SYNTHETIC_CLEAN_DATASET_PATH
                test_ratio = 0.0
            else:
                raise ValueError(f"Dataset {dataset} not supported")
            _prepare_ocr_fine_tuning_dataset(
                dataset=dataset, ocr_file=ocr_file, clean_file=clean_file, test_ratio=test_ratio
            )
        # add dataset to LLaMA-Factory dataset_info.json
        lf_dataset_info_path = BASE_DIR / "LLaMA-Factory" / "data" / "dataset_info.json"
        # Read the existing content
        with open(lf_dataset_info_path, "r", encoding="utf-8") as f:
            lf_dataset_info = json.load(f)
        # Add the new dataset info
        train_file_path = SFT_DATASET_DIR / "llama-factory" / f"{dataset.value}.json"
        lf_dataset_info[dataset.value] = {"file_name": str(train_file_path)}
        # Write back the updated content
        with open(lf_dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(lf_dataset_info, f, indent=2)
