# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.5.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

# Integrations: this needs to come before other ml imports
# in order to allow any 3rd-party code to initialize properly
from xlibs.integrations import (  # isort:skip
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)

# Configurations
from xlibs.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from xlibs.configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, AutoConfig
from xlibs.configuration_bart import BartConfig
from xlibs.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from xlibs.configuration_bert_generation import BertGenerationConfig
from xlibs.configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from xlibs.configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from xlibs.configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from xlibs.configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from xlibs.configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from xlibs.configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
from xlibs.configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from xlibs.configuration_encoder_decoder import EncoderDecoderConfig
from xlibs.configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from xlibs.configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
from xlibs.configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from xlibs.configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from xlibs.configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from xlibs.configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from xlibs.configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from xlibs.configuration_marian import MarianConfig
from xlibs.configuration_mbart import MBartConfig
from xlibs.configuration_mmbt import MMBTConfig
from xlibs.configuration_mobilebert import MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, MobileBertConfig
from xlibs.configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from xlibs.configuration_pegasus import PegasusConfig
from xlibs.configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from xlibs.configuration_rag import RagConfig
from xlibs.configuration_reformer import REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, ReformerConfig
from xlibs.configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from xlibs.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from xlibs.configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from xlibs.configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from xlibs.configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from xlibs.configuration_utils import PretrainedConfig
from xlibs.configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from xlibs.configuration_xlm_prophetnet import XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMProphetNetConfig
from xlibs.configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from xlibs.configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig
from xlibs.data import (
    DataProcessor,
    InputExample,
    InputFeatures,
    SingleSentenceClassificationProcessor,
    SquadExample,
    SquadFeatures,
    SquadV1Processor,
    SquadV2Processor,
    glue_compute_metrics,
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    squad_convert_examples_to_features,
    xnli_compute_metrics,
    xnli_output_modes,
    xnli_processors,
    xnli_tasks_num_labels,
)

# Files and general utilities
from xlibs.file_utils import (
    CONFIG_NAME,
    MODEL_CARD_NAME,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    SPIECE_UNDERLINE,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    TRANSFORMERS_CACHE,
    WEIGHTS_NAME,
    add_end_docstrings,
    add_start_docstrings,
    cached_path,
    is_apex_available,
    is_datasets_available,
    is_faiss_available,
    is_flax_available,
    is_psutil_available,
    is_py3nvml_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_tpu_available,
)
from xlibs.hf_argparser import HfArgumentParser

# Model Cards
from xlibs.modelcard import ModelCard

# TF 2.0 <=> PyTorch conversion utilities

# Pipelines
from xlibs.pipelines import (
    Conversation,
    ConversationalPipeline,
    CsvPipelineDataFormat,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    JsonPipelineDataFormat,
    NerPipeline,
    PipedPipelineDataFormat,
    Pipeline,
    PipelineDataFormat,
    QuestionAnsweringPipeline,
    SummarizationPipeline,
    Text2TextGenerationPipeline,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
    TranslationPipeline,
    ZeroShotClassificationPipeline,
    pipeline,
)

# Retriever
from xlibs.retrieval_rag import RagRetriever

# Tokenizers
from xlibs.tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer
from xlibs.tokenization_bart import BartTokenizer
from xlibs.tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer
from xlibs.tokenization_bert_japanese import BertJapaneseTokenizer, CharacterTokenizer, MecabTokenizer
from xlibs.tokenization_bertweet import BertweetTokenizer
from xlibs.tokenization_blenderbot import BlenderbotSmallTokenizer, BlenderbotTokenizer
from xlibs.tokenization_ctrl import CTRLTokenizer
from xlibs.tokenization_deberta import DebertaTokenizer
from xlibs.tokenization_distilbert import DistilBertTokenizer
from xlibs.tokenization_dpr import (
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer,
    DPRReaderOutput,
    DPRReaderTokenizer,
)
from xlibs.tokenization_electra import ElectraTokenizer
from xlibs.tokenization_flaubert import FlaubertTokenizer
from xlibs.tokenization_fsmt import FSMTTokenizer
from xlibs.tokenization_funnel import FunnelTokenizer
from xlibs.tokenization_gpt2 import GPT2Tokenizer
from xlibs.tokenization_herbert import HerbertTokenizer
from xlibs.tokenization_layoutlm import LayoutLMTokenizer
from xlibs.tokenization_longformer import LongformerTokenizer
from xlibs.tokenization_lxmert import LxmertTokenizer
from xlibs.tokenization_mobilebert import MobileBertTokenizer
from xlibs.tokenization_openai import OpenAIGPTTokenizer
from xlibs.tokenization_phobert import PhobertTokenizer
from xlibs.tokenization_prophetnet import ProphetNetTokenizer
from xlibs.tokenization_rag import RagTokenizer
from xlibs.tokenization_retribert import RetriBertTokenizer
from xlibs.tokenization_roberta import RobertaTokenizer
from xlibs.tokenization_squeezebert import SqueezeBertTokenizer
from xlibs.tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer
from xlibs.tokenization_utils import PreTrainedTokenizer
from xlibs.tokenization_utils_base import (
    AddedToken,
    BatchEncoding,
    CharSpan,
    PreTrainedTokenizerBase,
    SpecialTokensMixin,
    TensorType,
    TokenSpan,
)
from xlibs.tokenization_xlm import XLMTokenizer


if is_sentencepiece_available():
    from xlibs.tokenization_albert import AlbertTokenizer
    from xlibs.tokenization_bert_generation import BertGenerationTokenizer
    from xlibs.tokenization_camembert import CamembertTokenizer
    from xlibs.tokenization_marian import MarianTokenizer
    from xlibs.tokenization_mbart import MBartTokenizer
    from xlibs.tokenization_pegasus import PegasusTokenizer
    from xlibs.tokenization_reformer import ReformerTokenizer
    from xlibs.tokenization_t5 import T5Tokenizer
    from xlibs.tokenization_xlm_prophetnet import XLMProphetNetTokenizer
    from xlibs.tokenization_xlm_roberta import XLMRobertaTokenizer
    from xlibs.tokenization_xlnet import XLNetTokenizer
else:
    from xlibs.utils.dummy_sentencepiece_objects import *

if is_tokenizers_available():
    from xlibs.tokenization_albert_fast import AlbertTokenizerFast
    from xlibs.tokenization_bart_fast import BartTokenizerFast
    from xlibs.tokenization_bert_fast import BertTokenizerFast
    from xlibs.tokenization_camembert_fast import CamembertTokenizerFast
    from xlibs.tokenization_distilbert_fast import DistilBertTokenizerFast
    from xlibs.tokenization_dpr_fast import (
        DPRContextEncoderTokenizerFast,
        DPRQuestionEncoderTokenizerFast,
        DPRReaderTokenizerFast,
    )
    from xlibs.tokenization_electra_fast import ElectraTokenizerFast
    from xlibs.tokenization_funnel_fast import FunnelTokenizerFast
    from xlibs.tokenization_gpt2_fast import GPT2TokenizerFast
    from xlibs.tokenization_herbert_fast import HerbertTokenizerFast
    from xlibs.tokenization_layoutlm_fast import LayoutLMTokenizerFast
    from xlibs.tokenization_longformer_fast import LongformerTokenizerFast
    from xlibs.tokenization_lxmert_fast import LxmertTokenizerFast
    from xlibs.tokenization_mbart_fast import MBartTokenizerFast
    from xlibs.tokenization_mobilebert_fast import MobileBertTokenizerFast
    from xlibs.tokenization_openai_fast import OpenAIGPTTokenizerFast
    from xlibs.tokenization_pegasus_fast import PegasusTokenizerFast
    from xlibs.tokenization_reformer_fast import ReformerTokenizerFast
    from xlibs.tokenization_retribert_fast import RetriBertTokenizerFast
    from xlibs.tokenization_roberta_fast import RobertaTokenizerFast
    from xlibs.tokenization_squeezebert_fast import SqueezeBertTokenizerFast
    from xlibs.tokenization_t5_fast import T5TokenizerFast
    from xlibs.tokenization_utils_fast import PreTrainedTokenizerFast
    from xlibs.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
    from xlibs.tokenization_xlnet_fast import XLNetTokenizerFast

    if is_sentencepiece_available():
        from xlibs.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, convert_slow_tokenizer
else:
    from xlibs.utils.dummy_tokenizers_objects import *

# Trainer
from xlibs.trainer_callback import (
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from xlibs.trainer_utils import EvalPrediction, EvaluationStrategy, set_seed
from xlibs.training_args import TrainingArguments
from xlibs.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Modeling
if is_torch_available():
    # Benchmarks
    from xlibs.benchmark.benchmark import PyTorchBenchmark
    from xlibs.benchmark.benchmark_args import PyTorchBenchmarkArguments
    from xlibs.data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithPadding,
        default_data_collator,
    )
    from xlibs.data.datasets import (
        GlueDataset,
        GlueDataTrainingArguments,
        LineByLineTextDataset,
        LineByLineWithRefDataset,
        LineByLineWithSOPTextDataset,
        SquadDataset,
        SquadDataTrainingArguments,
        TextDataset,
        TextDatasetForNextSentencePrediction,
    )
    from xlibs.generation_beam_search import BeamScorer, BeamSearchScorer
    from xlibs.generation_logits_process import (
        LogitsProcessor,
        LogitsProcessorList,
        LogitsWarper,
        MinLengthLogitsProcessor,
        NoBadWordsLogitsProcessor,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
    from xlibs.generation_utils import top_k_top_p_filtering
    from xlibs.modeling_albert import (
        ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        AlbertForMaskedLM,
        AlbertForMultipleChoice,
        AlbertForPreTraining,
        AlbertForQuestionAnswering,
        AlbertForSequenceClassification,
        AlbertForTokenClassification,
        AlbertModel,
        AlbertPreTrainedModel,
        load_tf_weights_in_albert,
    )
    from xlibs.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
        MODEL_WITH_LM_HEAD_MAPPING,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForMultipleChoice,
        AutoModelForNextSentencePrediction,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
    )
    from xlibs.modeling_bart import (
        BART_PRETRAINED_MODEL_ARCHIVE_LIST,
        BartForConditionalGeneration,
        BartForQuestionAnswering,
        BartForSequenceClassification,
        BartModel,
        PretrainedBartModel,
    )
    from xlibs.modeling_bert import (
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForNextSentencePrediction,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertLayer,
        BertLMHeadModel,
        BertModel,
        BertPreTrainedModel,
        load_tf_weights_in_bert,
    )
    from xlibs.modeling_bert_generation import (
        BertGenerationDecoder,
        BertGenerationEncoder,
        load_tf_weights_in_bert_generation,
    )
    from xlibs.modeling_blenderbot import BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotForConditionalGeneration
    from xlibs.modeling_camembert import (
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        CamembertForCausalLM,
        CamembertForMaskedLM,
        CamembertForMultipleChoice,
        CamembertForQuestionAnswering,
        CamembertForSequenceClassification,
        CamembertForTokenClassification,
        CamembertModel,
    )
    from xlibs.modeling_ctrl import CTRL_PRETRAINED_MODEL_ARCHIVE_LIST, CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel
    from xlibs.modeling_deberta import (
        DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        DebertaForSequenceClassification,
        DebertaModel,
        DebertaPreTrainedModel,
    )
    from xlibs.modeling_distilbert import (
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        DistilBertForMaskedLM,
        DistilBertForMultipleChoice,
        DistilBertForQuestionAnswering,
        DistilBertForSequenceClassification,
        DistilBertForTokenClassification,
        DistilBertModel,
        DistilBertPreTrainedModel,
    )
    from xlibs.modeling_dpr import (
        DPRContextEncoder,
        DPRPretrainedContextEncoder,
        DPRPretrainedQuestionEncoder,
        DPRPretrainedReader,
        DPRQuestionEncoder,
        DPRReader,
    )
    from xlibs.modeling_electra import (
        ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST,
        ElectraForMaskedLM,
        ElectraForMultipleChoice,
        ElectraForPreTraining,
        ElectraForQuestionAnswering,
        ElectraForSequenceClassification,
        ElectraForTokenClassification,
        ElectraModel,
        ElectraPreTrainedModel,
        load_tf_weights_in_electra,
    )
    from xlibs.modeling_encoder_decoder import EncoderDecoderModel
    from xlibs.modeling_flaubert import (
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        FlaubertForMultipleChoice,
        FlaubertForQuestionAnswering,
        FlaubertForQuestionAnsweringSimple,
        FlaubertForSequenceClassification,
        FlaubertForTokenClassification,
        FlaubertModel,
        FlaubertWithLMHeadModel,
    )
    from xlibs.modeling_fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
    from xlibs.modeling_funnel import (
        FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST,
        FunnelBaseModel,
        FunnelForMaskedLM,
        FunnelForMultipleChoice,
        FunnelForPreTraining,
        FunnelForQuestionAnswering,
        FunnelForSequenceClassification,
        FunnelForTokenClassification,
        FunnelModel,
        load_tf_weights_in_funnel,
    )
    from xlibs.modeling_gpt2 import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        GPT2DoubleHeadsModel,
        GPT2ForSequenceClassification,
        GPT2LMHeadModel,
        GPT2Model,
        GPT2PreTrainedModel,
        load_tf_weights_in_gpt2,
    )
    from xlibs.modeling_layoutlm import (
        LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        LayoutLMForMaskedLM,
        LayoutLMForTokenClassification,
        LayoutLMModel,
    )
    from xlibs.modeling_longformer import (
        LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        LongformerForMaskedLM,
        LongformerForMultipleChoice,
        LongformerForQuestionAnswering,
        LongformerForSequenceClassification,
        LongformerForTokenClassification,
        LongformerModel,
        LongformerSelfAttention,
    )
    from xlibs.modeling_lxmert import (
        LxmertEncoder,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
        LxmertPreTrainedModel,
        LxmertVisualFeatureEncoder,
        LxmertXLayer,
    )
    from xlibs.modeling_marian import MarianMTModel
    from xlibs.modeling_mbart import MBartForConditionalGeneration
    from xlibs.modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings
    from xlibs.modeling_mobilebert import (
        MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        MobileBertForMaskedLM,
        MobileBertForMultipleChoice,
        MobileBertForNextSentencePrediction,
        MobileBertForPreTraining,
        MobileBertForQuestionAnswering,
        MobileBertForSequenceClassification,
        MobileBertForTokenClassification,
        MobileBertLayer,
        MobileBertModel,
        MobileBertPreTrainedModel,
        load_tf_weights_in_mobilebert,
    )
    from xlibs.modeling_openai import (
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
        OpenAIGPTDoubleHeadsModel,
        OpenAIGPTForSequenceClassification,
        OpenAIGPTLMHeadModel,
        OpenAIGPTModel,
        OpenAIGPTPreTrainedModel,
        load_tf_weights_in_openai_gpt,
    )
    from xlibs.modeling_pegasus import PegasusForConditionalGeneration
    from xlibs.modeling_prophetnet import (
        PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        ProphetNetDecoder,
        ProphetNetEncoder,
        ProphetNetForCausalLM,
        ProphetNetForConditionalGeneration,
        ProphetNetModel,
        ProphetNetPreTrainedModel,
    )
    from xlibs.modeling_rag import RagModel, RagSequenceForGeneration, RagTokenForGeneration
    from xlibs.modeling_reformer import (
        REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
        ReformerAttention,
        ReformerForMaskedLM,
        ReformerForQuestionAnswering,
        ReformerForSequenceClassification,
        ReformerLayer,
        ReformerModel,
        ReformerModelWithLMHead,
    )
    from xlibs.modeling_retribert import RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST, RetriBertModel, RetriBertPreTrainedModel
    from xlibs.modeling_roberta import (
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        RobertaForCausalLM,
        RobertaForMaskedLM,
        RobertaForMultipleChoice,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaForTokenClassification,
        RobertaModel,
    )
    from xlibs.modeling_squeezebert import (
        SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        SqueezeBertForMaskedLM,
        SqueezeBertForMultipleChoice,
        SqueezeBertForQuestionAnswering,
        SqueezeBertForSequenceClassification,
        SqueezeBertForTokenClassification,
        SqueezeBertModel,
        SqueezeBertModule,
        SqueezeBertPreTrainedModel,
    )
    from xlibs.modeling_t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )
    from xlibs.modeling_transfo_xl import (
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST,
        AdaptiveEmbedding,
        TransfoXLLMHeadModel,
        TransfoXLModel,
        TransfoXLPreTrainedModel,
        load_tf_weights_in_transfo_xl,
    )
    from xlibs.modeling_utils import Conv1D, PreTrainedModel, apply_chunking_to_forward, prune_layer
    from xlibs.modeling_xlm import (
        XLM_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMForMultipleChoice,
        XLMForQuestionAnswering,
        XLMForQuestionAnsweringSimple,
        XLMForSequenceClassification,
        XLMForTokenClassification,
        XLMModel,
        XLMPreTrainedModel,
        XLMWithLMHeadModel,
    )
    from xlibs.modeling_xlm_prophetnet import (
        XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMProphetNetDecoder,
        XLMProphetNetEncoder,
        XLMProphetNetForCausalLM,
        XLMProphetNetForConditionalGeneration,
        XLMProphetNetModel,
    )
    from xlibs.modeling_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLMRobertaForCausalLM,
        XLMRobertaForMaskedLM,
        XLMRobertaForMultipleChoice,
        XLMRobertaForQuestionAnswering,
        XLMRobertaForSequenceClassification,
        XLMRobertaForTokenClassification,
        XLMRobertaModel,
    )
    from xlibs.modeling_xlnet import (
        XLNET_PRETRAINED_MODEL_ARCHIVE_LIST,
        XLNetForMultipleChoice,
        XLNetForQuestionAnswering,
        XLNetForQuestionAnsweringSimple,
        XLNetForSequenceClassification,
        XLNetForTokenClassification,
        XLNetLMHeadModel,
        XLNetModel,
        XLNetPreTrainedModel,
        load_tf_weights_in_xlnet,
    )

    # Optimization
    from xlibs.optimization import (
        Adafactor,
        AdamW,
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

    # Trainer
    from xlibs.trainer import Trainer
    from xlibs.trainer_pt_utils import torch_distributed_zero_first
else:
    from xlibs.utils.dummy_pt_objects import *


if is_flax_available():
    from xlibs.modeling_flax_bert import FlaxBertModel
    from xlibs.modeling_flax_roberta import FlaxRobertaModel
else:
    # Import the same objects as dummies to get them in the namespace.
    # They will raise an import error if the user tries to instantiate / use them.
    from xlibs.utils.dummy_flax_objects import *


if not is_tf_available() and not is_torch_available():
    logger.warning(
        "Neither PyTorch nor TensorFlow >= 2.0 have been found."
        "Models won't be available and only tokenizers, configuration"
        "and file/data utilities can be used."
    )
