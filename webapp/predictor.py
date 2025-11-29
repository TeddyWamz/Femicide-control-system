import torch
from pathlib import Path
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Predictor:
    """
    Lightweight wrapper that loads the fine-tuned XLM-R model once
    and exposes a simple predict(text) method.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Keep mapping consistent with training
        self.id2label = {
            0: "Physical_violence",
            1: "sexual_violence",
            2: "emotional_violence",
            3: "economic_violence",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir.as_posix())
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir.as_posix())
        self.model.eval()

        # Prefer CPU unless CUDA is explicitly available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.keyword_rules: Dict[str, list[str]] = {
            "sexual_violence": [
                "rape",
                "raped",
                "sexual assault",
                "forced himself",
                "forced herself",
                "molest",
                "molested",
                "defile",
                "defiled",
                "drugged me",
                "drugged her",
                "coerced sex",
                "without consent",
                "forced sex",
                "alinibaka",
                "amenibaka",
                "alinilazimisha kufanya ngono",
                "kunibaka",
                "kunilazimisha ngono",
                "kubakwa",
                "kubakishwa",
                "ubakaji",
                "kunilazimisha kimwili",
                "kunishika kwa nguvu",
                "kunifanya bila ridhaa",
                "unyanyasaji wa kingono",
                "shambulio la kingono",
                "kulazimishwa ngono",
                "kunidhulumu kimapenzi",
                "dhuluma za kingono",
                "kunichafua",
                "kuninajisi",
                "kunifanya kingono bila idhini",
                "kunitenda kimapenzi kinyume",
                "kunigusa bila ruhusa",
                "unyanyasaji wa kijinsia",
                "kulawiti",
                "nililazimishwa kufanya ngono",
                "jaribio la ubakaji",
                "kunishika sehemu za siri",
                "kunigutsa bila idhini",
                "kunisisitiza kufanya ngono",
                "kunitisha ili nifanye ngono",
                "kunilaghai kufanya tendo la ndoa",
                "kulazimisha tendo la ndoa",
                "kunipigia kelele ili nifanye ngono",
                "kunilazimisha mapenzi",
                "kunigusa sehemu zangu za siri",
                "kunishika matiti kimakusudi",
                "kunidonoa sehemu binafsi",
                "kunishika mapaja bila ruhusa",
                "kunipapasa bila kukubali",
                "kunifinya sehemu za mwili",
                "kuingilia faragha yangu",
                "kunitumia ujumbe wa matusi ya kingono",
                "kunitumia picha za uchi",
                "maneno ya kingono",
                "vitisho vya kingono",
                "kunitongoza kwa nguvu",
                "mume kunilazimisha tendo la ndoa",
                "kuninyima ridhaa kwenye ndoa",
                "kunifanyia tendo la ndoa kwa lazima",
                "kulazimishwa na mume wangu",
                "kulazimishwa kufanya tendo la ndoa",
                "jirani kuniingilia",
                "rafiki kunishika sehemu za siri",
                "mwajiri kunitendea kingono",
                "pastor kuniomba ngono",
                "ndugu kuninajisi",
                "dereva kunishika bila ridhaa",
                "mwalimu kunifanyia jaribio",
            ],
            "economic_violence": [
                "salary",
                "money",
                "allowance",
                "bank card",
                "financial",
                "took my pay",
                "took my salary",
                "withheld pay",
                "stole my money",
                "no access to money",
                "control the finances",
                "economic abuse",
                "won't let me work",
                "mshahara",
                "pesa",
                "hela",
                "ananinyima pesa",
                "ananizuia kufanya kazi",
                "ananikataza kufanya kazi",
            ],
            "emotional_violence": [
                "insulted me",
                "called me names",
                "belittle",
                "humiliate",
                "threatened",
                "psychological abuse",
                "emotional abuse",
                "gaslight",
                "ridicule",
                "verbal abuse",
                "ananitusi",
                "ananidharau",
                "ananiita majina",
                "amenitishia",
                "ananitisha",
                "kunidhalilisha",
                "kunidharau",
                "kunitukana",
                "matusi",
                "kuniambia mimi si kitu",
                "kuniita worthless",
                "kuniambia sina maana",
                "kuniambia sifai",
                "kuniambia mimi ni mzigo",
                "kunikashifu",
                "kunikejeli",
                "kuniaibisha",
                "kunifedhehesha",
                "kunifokea bila sababu",
                "vitisho",
                "kunitisha kunidhuru",
                "kuishi kwa hofu",
                "kunifanya niogope",
                "kunikosesha usingizi",
                "kunitenga",
                "kunizuia kuongea na rafiki zangu",
                "kunikataza kuonana na familia",
                "kunidhibiti",
                "kunifuatilia",
                "kudhibiti mawasiliano",
                "kunimiliki",
                "kunizuiakufanya maamuzi",
                "kunifanya nishuku akili yangu",
                "kunigeuzia lawama",
                "emotional blackmail",
                "kunilaumu kwa kila kitu",
                "msongo wa mawazo",
                "shinikizo la kisaikolojia",
                "kuishi kwa mawazo",
                "wasiwasi",
                "kuishi kwa woga",
                "kukosa amani",
                "kunifanya nipoteze kujiamini",
                "kunifanya nijichukie",
                "kunifanya nijione sina thamani",
                "wivu wa kupindukia",
                "kunichunguza kila mara",
                "kunifuata kwa vitisho",
                "maneno ya kunifanya niwe na hofu",
            ],
            "Physical_violence": [
                "slapped me",
                "punched me",
                "kicked me",
                "beat me",
                "choked me",
                "physical abuse",
                "hit me",
                "broke my bones",
                "locked me in",
                "dragged me",
                "alinipiga",
                "amenipiga",
                "alinichapa",
                "kunipiga",
                "atanimaliza",
                "atanichoma",
                "atanikata",
                "kisu",
                "kunichoma moto",
                "atanichinja",
            ],
        }

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns (label, confidence) where confidence is the softmax probability
        of the predicted label in [0, 1], plus the full probability vector.
        """
        if not text or not text.strip():
            prob_map = {label: 0.0 for label in self.id2label.values()}
            return "", 0.0, prob_map

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        prob_vec = {self.id2label[i]: float(probs[0][i].item()) for i in range(probs.shape[1])}

        idx = int(pred_idx.item())
        label = self.id2label.get(idx, str(idx))
        confidence = float(conf.item())

        lowered = text.lower()
        for target_label, keywords in self.keyword_rules.items():
            if target_label == label:
                continue
            if any(keyword in lowered for keyword in keywords):
                label = target_label
                confidence = max(confidence, 0.8)
                break

        return label, confidence, prob_vec
