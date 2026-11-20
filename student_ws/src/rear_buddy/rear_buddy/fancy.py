"""Which kind of car is behind us? A check takes (frame, detection) and returns (tier, brand):

    tier   'premium' -> job pitch, 'budget' -> "old is gold", None -> normal hello
    brand  what was recognised ('BMW', 'a small cheap city car', ...) or None

Checks (make_fancy_check):
    clip    recognise the brand with CLIP (open_clip, ViT-L-14), zero-shot: no training, runs offline once the
            weights are downloaded (~1.7 GB, first run only). Brand lists: CAR_BRANDS in messages.py.
    always  every car is premium (demo the job pitch with any car photo)
    budget  every car is budget
    never   every car is an ordinary car

Only the closest vehicle is checked (pipeline.analyse), so the cost is one small image per frame.
"""
from .messages import BUDGET_DESCRIPTIONS, CAR_BRANDS

MIN_CROP_PX = 32  # boxes smaller than this are too far away to recognise a brand


class FixedTier:
    """Always the same answer, for testing and demos."""

    def __init__(self, tier):
        self.tier = tier

    def __call__(self, frame, detection):
        return self.tier, None


def labels_from(brands=None, descriptions=None):
    """[(label, tier, prompts)] for the brand lists: brands get several phrasings, descriptions one."""
    brands = CAR_BRANDS if brands is None else brands
    descriptions = BUDGET_DESCRIPTIONS if descriptions is None else descriptions
    labels = []
    for tier, names in brands.items():
        for name in names:
            labels.append((name, None if tier == 'other' else tier,
                           [f'a photo of a {name} car', f'a {name}', f'the front of a {name}']))
    labels += [(d, 'budget', [f'a photo of {d}']) for d in descriptions]
    return labels


def decide(probs, labels, min_conf=0.5):
    """(tier, brand) from per-label probabilities: the best label, unless the model is unsure."""
    best = max(range(len(labels)), key=lambda i: probs[i])
    if probs[best] < min_conf:
        return None, None
    return labels[best][1], labels[best][0]


class ClipBrandCheck:
    def __init__(self, model='ViT-L-14', pretrained='laion2b_s32b_b82k', min_conf=0.5, brands=None,
                 descriptions=None, device=None):
        import open_clip
        import torch
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available()
                                 else 'cuda' if torch.cuda.is_available() else 'cpu')
        # Downloads on the first run, afterwards from the local cache (also without internet, e.g. on the car's Wi-Fi).
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained,
                                                                               device=self.device)
        self.model.eval()
        self.min_conf = min_conf
        self.labels = labels_from(brands, descriptions)
        tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            per_label = []
            for _, _, prompts in self.labels:
                f = self.model.encode_text(tokenizer(prompts).to(self.device))
                per_label.append((f / f.norm(dim=-1, keepdim=True)).mean(0))
            text = torch.stack(per_label)
            self.text = text / text.norm(dim=-1, keepdim=True)

    def __call__(self, frame, detection):
        from PIL import Image
        import cv2
        car = crop(frame, detection)
        if min(car.shape[:2]) < MIN_CROP_PX:
            return None, None
        image = self.preprocess(Image.fromarray(cv2.cvtColor(car, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(image)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = (100 * f @ self.text.T).softmax(-1)[0].cpu().tolist()
        return decide(probs, self.labels, self.min_conf)


def make_fancy_check(name, log=print):
    if name == 'clip':
        try:
            log('loading the car brand model (CLIP ViT-L-14) ...')
            return ClipBrandCheck()
        except Exception as e:  # noqa: BLE001 - missing package or weights: keep going without brands
            log(f'car brand check unavailable ({e}); every car gets the normal hello. pip install open_clip_torch')
            return FixedTier(None)
    fixed = {'always': 'premium', 'budget': 'budget', 'never': None}
    if name not in fixed:
        raise ValueError(f'unknown car check {name!r}, choose from clip, {", ".join(fixed)}')
    return FixedTier(fixed[name])


def crop(frame, detection):
    """The detection's box cut out of the frame, for checks that look at the car itself."""
    x1, y1, x2, y2 = (int(v) for v in detection.box)
    return frame[max(y1, 0):y2, max(x1, 0):x2]
