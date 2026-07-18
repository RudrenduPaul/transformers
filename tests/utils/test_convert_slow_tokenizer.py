import unittest
import warnings
from dataclasses import dataclass

from transformers.convert_slow_tokenizer import LlamaConverter, SpmConverter
from transformers.testing_utils import get_tests_dir, require_sentencepiece


@dataclass
class FakeOriginalTokenizer:
    vocab_file: str


class ConvertSlowTokenizerTest(unittest.TestCase):
    def test_spm_converter_bytefallback_warning(self):
        spm_model_file_without_bytefallback = get_tests_dir("fixtures/test_sentencepiece.model")
        spm_model_file_with_bytefallback = get_tests_dir("fixtures/test_sentencepiece_with_bytefallback.model")

        original_tokenizer_without_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_without_bytefallback)

        with warnings.catch_warnings(record=True) as w:
            _ = SpmConverter(original_tokenizer_without_bytefallback)
        # We are looking for if there is any `UserWarning` with
        # `The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.`
        w = [x for x in w if x.category.__name__ != "DeprecationWarning"]
        self.assertEqual(len(w), 0)

        original_tokenizer_with_bytefallback = FakeOriginalTokenizer(vocab_file=spm_model_file_with_bytefallback)

        with warnings.catch_warnings(record=True) as w:
            _ = SpmConverter(original_tokenizer_with_bytefallback)
        w = [x for x in w if x.category.__name__ != "DeprecationWarning"]
        self.assertEqual(len(w), 1)

        self.assertIn(
            "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
            " which is not implemented in the fast tokenizers.",
            str(w[0].message),
        )

    @require_sentencepiece
    def test_spm_converter_accepts_raw_sentencepiece_processor(self):
        """
        Regression test for #28370: `Converter` subclasses should also accept a raw
        `sentencepiece.SentencePieceProcessor` (e.g. loaded directly by the user from a custom
        `.model` file) rather than requiring a slow tokenizer instance with a `vocab_file`
        attribute.
        """
        import sentencepiece as spm

        spm_model_file = get_tests_dir("fixtures/test_sentencepiece.model")

        original_tokenizer = spm.SentencePieceProcessor()
        original_tokenizer.Load(spm_model_file)
        self.assertFalse(hasattr(original_tokenizer, "vocab_file"))

        fast_tokenizer = LlamaConverter(original_tokenizer).converted()
        vocab = fast_tokenizer.get_vocab()

        # There is no HF added-tokens layer to consult for a raw `SentencePieceProcessor`, so the
        # first three ids should fall back to the raw SentencePiece model's own special-token
        # pieces.
        for token_id in range(3):
            piece = original_tokenizer.id_to_piece(token_id)
            self.assertEqual(vocab[piece], token_id)

        self.assertEqual(fast_tokenizer.get_vocab_size(), original_tokenizer.GetPieceSize())
