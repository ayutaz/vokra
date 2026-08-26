import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("hf_mac_coverage.py")
SPEC = importlib.util.spec_from_file_location("hf_mac_coverage", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class HfMacCoverageTest(unittest.TestCase):
    def test_readme_architecture_accepts_generated_card_shape(self):
        self.assertEqual(
            audit.parse_readme_architecture("| Architecture | `moonshine` |\n"),
            "moonshine",
        )
        self.assertEqual(
            audit.parse_readme_architecture("| Architecture | voxtral |\n"),
            "voxtral",
        )
        self.assertIsNone(audit.parse_readme_architecture("# no table\n"))

    def test_engine_parser_separates_routed_and_bound(self):
        source = '''
const ARCH_WHISPER: &str = "whisper";
const ARCH_MOONSHINE: &str = "moonshine";
const BOUND_ARCHES: &[BoundArch] = &[
    BoundArch { arch: "snac", reason: "partial" },
];
'''
        routed, bound = audit.parse_engine_arches(source)
        self.assertEqual(routed, {"whisper", "moonshine"})
        self.assertEqual(bound, {"snac"})

    def test_classification_never_turns_partial_into_metal(self):
        routed = {
            "audiobox-aesthetics",
            "audioseal_real_weight",
            "ast",
            "canary",
            "canary-1b-flash",
            "whisper",
            "silero-vad",
            "magnet_small_10secs",
            "csm",
            "dac",
            "nsnet2",
            "pyannote-segmentation",
            "pyannote-speaker-diarization",
            "rmvpe",
            "sbv2",
            "vocos",
            "fsmn-vad",
            "firered_vad",
            "focalcodec",
            "funcodec",
            "speechtokenizer",
            "ten_vad",
            "rnnoise",
            "snac",
            "nkf_aec",
            "xvector",
            "ecapa_tdnn",
            "wespeaker",
            "titanet-large",
            "denoise",
            "dnsmos",
            "nisqa_v2_weight",
            "emotion2vec",
            "utmos",
            "metricgan_plus",
            "frcrn",
            "miocodec",
            "mossformer2_ss_16k",
            "moss_audio_tokenizer",
            "moss_tts",
            "musicgen",
            "tiger_separator",
            "deepfake_detection",
        }
        bound = set()
        full = audit.RepoRecord("vokra/whisper", "abc", ("model.gguf",), "whisper")
        ast = audit.RepoRecord("vokra/ast", "abc", ("ast.gguf",), "ast")
        audiobox = audit.RepoRecord(
            "vokra/audiobox-aesthetics",
            "abc",
            ("audiobox-aesthetics.gguf",),
            "audiobox-aesthetics",
        )
        audioseal = audit.RepoRecord(
            "vokra/audioseal-real-weight",
            "abc",
            ("audioseal-real-weight.gguf",),
            "audioseal_real_weight",
        )
        canary_flash = audit.RepoRecord(
            "vokra/canary-1b-flash",
            "abc",
            ("canary-1b-flash.gguf",),
            "canary-1b-flash",
        )
        canary_v2 = audit.RepoRecord(
            "vokra/canary-1b-v2",
            "abc",
            ("canary.gguf", "model.gguf"),
            "canary",
        )
        corrected_canary_v2 = audit.RepoRecord(
            "vokra/canary-1b-v2-corrected",
            "abc",
            ("canary.gguf",),
            "canary",
        )
        corrected_canary_flash = audit.RepoRecord(
            "vokra/canary-1b-flash-corrected",
            "abc",
            ("canary-1b-flash.gguf",),
            "canary-1b-flash",
        )
        miocodec = audit.RepoRecord(
            "vokra/miocodec-25hz-44khz-v2",
            "abc",
            ("miocodec-25hz-44khz-v2.gguf",),
            "miocodec",
        )
        funcodec = audit.RepoRecord(
            "vokra/funcodec", "abc", ("model.gguf",), "funcodec"
        )
        speechtokenizer = audit.RepoRecord(
            "vokra/speechtokenizer", "abc", ("model.gguf",), "speechtokenizer"
        )
        musicgen_small = audit.RepoRecord(
            "vokra/musicgen-small", "abc", ("model.gguf",), "musicgen"
        )
        moss_full = audit.RepoRecord(
            "vokra/moss-audio-tokenizer-full",
            "abc",
            ("moss-audio-tokenizer-full.gguf",),
            "moss_audio_tokenizer",
        )
        moss_nano = audit.RepoRecord(
            "vokra/moss-audio-tokenizer-nano",
            "abc",
            ("moss-audio-tokenizer-nano.gguf",),
            "moss_audio_tokenizer",
        )
        corrected_moss_nano = audit.RepoRecord(
            "vokra/moss-audio-tokenizer-nano-corrected",
            "abc",
            ("moss-audio-tokenizer-nano.gguf",),
            "moss_audio_tokenizer",
        )
        moss_delay = audit.RepoRecord(
            "vokra/moss-tts", "abc", ("moss-tts.gguf",), "moss_tts"
        )
        moss_local = audit.RepoRecord(
            "vokra/moss-tts-local-transformer-v1.5",
            "abc",
            ("moss-tts-local.gguf",),
            "moss_tts",
        )
        moss_audio_instruct = audit.RepoRecord(
            "vokra/moss-audio-4b-instruct",
            "abc",
            ("moss-audio-4b.gguf",),
            "moss_tts",
        )
        musicgen_medium = audit.RepoRecord(
            "vokra/musicgen-medium", "abc", ("model.gguf",), "musicgen"
        )
        audiogen_medium = audit.RepoRecord(
            "vokra/audiogen-medium", "abc", ("model.gguf",), "musicgen"
        )
        silero = audit.RepoRecord(
            "vokra/silero", "abc", ("model.gguf",), "silero-vad"
        )
        dac = audit.RepoRecord("vokra/dac-44khz", "abc", ("model.gguf",), "dac")
        snac = audit.RepoRecord("vokra/snac", "abc", ("model.gguf",), "snac")
        routed_partial = audit.RepoRecord(
            "vokra/magnet", "abc", ("model.gguf",), "magnet_small_10secs"
        )
        csm = audit.RepoRecord("vokra/csm", "abc", ("model.gguf",), "csm")
        sbv2 = audit.RepoRecord("vokra/sbv2", "abc", ("model.gguf",), "sbv2")
        nsnet2 = audit.RepoRecord(
            "vokra/nsnet2", "abc", ("model.gguf",), "nsnet2"
        )
        pyannote = audit.RepoRecord(
            "vokra/pyannote", "abc", ("model.gguf",), "pyannote-segmentation"
        )
        pyannote_diarization = audit.RepoRecord(
            "vokra/pyannote-speaker-diarization-3.1",
            "abc",
            ("pipeline.gguf",),
            "pyannote-speaker-diarization",
        )
        rmvpe = audit.RepoRecord("vokra/rmvpe", "abc", ("model.gguf",), "rmvpe")
        vocos = audit.RepoRecord("vokra/vocos", "abc", ("model.gguf",), "vocos")
        fsmn = audit.RepoRecord("vokra/fsmn", "abc", ("model.gguf",), "fsmn-vad")
        firered = audit.RepoRecord(
            "vokra/firered", "abc", ("model.gguf",), "firered_vad"
        )
        focalcodec = audit.RepoRecord(
            "vokra/focalcodec-50hz", "abc", ("model.gguf",), "focalcodec"
        )
        ten_vad = audit.RepoRecord("vokra/ten-vad", "abc", ("model.gguf",), "ten_vad")
        rnnoise = audit.RepoRecord("vokra/rnnoise", "abc", ("model.gguf",), "rnnoise")
        nkf_aec = audit.RepoRecord("vokra/nkf-aec", "abc", ("model.gguf",), "nkf_aec")
        xvector = audit.RepoRecord("vokra/xvector", "abc", ("model.gguf",), "xvector")
        ecapa = audit.RepoRecord("vokra/ecapa", "abc", ("model.gguf",), "ecapa_tdnn")
        wespeaker = audit.RepoRecord(
            "vokra/pyannote-wespeaker-voxceleb-resnet34-lm",
            "abc",
            ("model.gguf",),
            "wespeaker",
        )
        bad_wespeaker = audit.RepoRecord(
            "vokra/wespeaker", "abc", ("model.gguf",), "wespeaker"
        )
        titanet = audit.RepoRecord(
            "vokra/titanet-l", "abc", ("model.gguf",), "titanet-large"
        )
        deepfilternet = audit.RepoRecord(
            "vokra/deepfilternet3", "abc", ("model.gguf",), "denoise"
        )
        utmos = audit.RepoRecord(
            "vokra/utmos22-strong", "abc", ("model.gguf",), "utmos"
        )
        dnsmos = audit.RepoRecord(
            "vokra/dnsmos-p808-p835", "abc", ("model.gguf",), "dnsmos"
        )
        metricgan = audit.RepoRecord(
            "vokra/metricgan-plus-voicebank",
            "abc",
            ("metricgan-plus.gguf",),
            "metricgan_plus",
        )
        nisqa = audit.RepoRecord(
            "vokra/nisqa-v2",
            "abc",
            ("nisqa.gguf",),
            "nisqa_v2_weight",
        )
        frcrn = audit.RepoRecord(
            "vokra/frcrn", "abc", ("frcrn.gguf",), "frcrn"
        )
        emotion2vec = audit.RepoRecord(
            "vokra/emotion2vec", "abc", ("model.gguf",), "emotion2vec"
        )
        deepfake = audit.RepoRecord(
            "vokra/deepfake-audio-detection-v2",
            "abc",
            ("deepfake-detection.gguf",),
            "deepfake_detection",
        )
        tiger = audit.RepoRecord(
            "vokra/tiger-dnr",
            "abc",
            ("tiger-dnr.gguf",),
            "tiger_separator",
        )
        mossformer2 = audit.RepoRecord(
            "vokra/mossformer2-ss-16k",
            "abc",
            ("mossformer2-ss-16k.gguf",),
            "mossformer2_ss_16k",
        )
        missing = audit.RepoRecord("vokra/other", "abc", ("model.gguf",), "other")
        bad_ecapa = audit.RepoRecord(
            "vokra/voice-gender-classifier", "abc", ("model.gguf",), "ecapa_tdnn"
        )
        corrupt_ecapa = audit.RepoRecord(
            "vokra/speechbrain-spkrec-ecapa-voxceleb",
            "abc",
            ("model.gguf",),
            "ecapa_tdnn",
        )
        empty = audit.RepoRecord("vokra/empty", "abc", (), None)

        self.assertEqual(audit.classify(full, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(ast, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(audiobox, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(audiobox, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(audioseal, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(audioseal, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(canary_flash, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(canary_flash, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(audit.classify(canary_v2, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(canary_v2, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(
            audit.classify(corrected_canary_v2, routed, bound).cpu_code, "full"
        )
        self.assertEqual(
            audit.classify(corrected_canary_v2, routed, bound).metal_code, "full"
        )
        self.assertEqual(
            audit.classify(corrected_canary_flash, routed, bound).cpu_code, "full"
        )
        self.assertEqual(
            audit.classify(corrected_canary_flash, routed, bound).metal_code, "full"
        )
        self.assertEqual(audit.classify(miocodec, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(miocodec, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(funcodec, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(funcodec, routed, bound).metal_code, "full")
        self.assertEqual(
            audit.classify(speechtokenizer, routed, bound).cpu_code, "full"
        )
        self.assertEqual(
            audit.classify(speechtokenizer, routed, bound).metal_code, "full"
        )
        self.assertEqual(audit.classify(musicgen_small, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(musicgen_small, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(moss_full, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(moss_full, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(moss_nano, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(moss_nano, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(
            audit.classify(corrected_moss_nano, routed, bound).cpu_code, "full"
        )
        self.assertEqual(
            audit.classify(corrected_moss_nano, routed, bound).metal_code, "full"
        )
        self.assertEqual(audit.classify(moss_delay, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(moss_delay, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(moss_local, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(moss_local, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(
            audit.classify(moss_audio_instruct, routed, bound).cpu_code,
            "no-runtime-binder",
        )
        self.assertEqual(audit.classify(musicgen_medium, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(audiogen_medium, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(silero, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(dac, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(snac, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(snac, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(routed_partial, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(csm, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(sbv2, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(nsnet2, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(nsnet2, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(audit.classify(pyannote, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(pyannote, routed, bound).metal_code, "full")
        self.assertEqual(
            audit.classify(pyannote_diarization, routed, bound).cpu_code, "full"
        )
        self.assertEqual(
            audit.classify(pyannote_diarization, routed, bound).metal_code, "full"
        )
        self.assertEqual(audit.classify(rmvpe, routed, bound).cpu_code, "partial")
        self.assertEqual(
            audit.classify(rmvpe, routed, bound).metal_code, "blocked-by-cpu"
        )
        self.assertEqual(audit.classify(vocos, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(fsmn, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(firered, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(focalcodec, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(ten_vad, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(rnnoise, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(nkf_aec, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(xvector, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(ecapa, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(wespeaker, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(bad_wespeaker, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(titanet, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(deepfilternet, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(deepfilternet, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(utmos, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(utmos, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(dnsmos, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(dnsmos, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(metricgan, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(metricgan, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(nisqa, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(nisqa, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(frcrn, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(frcrn, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(emotion2vec, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(emotion2vec, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(deepfake, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(deepfake, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(tiger, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(tiger, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(mossformer2, routed, bound).cpu_code, "full")
        self.assertEqual(audit.classify(mossformer2, routed, bound).metal_code, "full")
        self.assertEqual(audit.classify(bad_ecapa, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(corrupt_ecapa, routed, bound).cpu_code, "partial")
        self.assertEqual(audit.classify(missing, routed, bound).cpu_code, "no-runtime-binder")
        self.assertEqual(audit.classify(empty, routed, bound).cpu_code, "not-artifact")

    def test_tsv_rows_match_header_width(self):
        record = audit.RepoRecord(
            "vokra/whisper", "abc", ("model.gguf",), "whisper"
        )
        lines = audit.render_tsv([record], {"whisper"}, set()).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0].split("\t")), len(lines[1].split("\t")))
        self.assertEqual(lines[1].split("\t")[5], "full")


if __name__ == "__main__":
    unittest.main()
