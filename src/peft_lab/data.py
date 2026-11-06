from __future__ import annotations

import random
import re
from dataclasses import dataclass

INTENTS = [
    "card_arrival",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "pending_transfer",
    "transfer_fee_charged",
    "card_payment_not_recognised",
    "declined_cash_withdrawal",
    "wrong_amount_of_cash_received",
]

FORMAL = {
    "card_arrival": [
        "When will my new card arrive", "Please provide the card delivery date", "My physical card has not arrived yet",
        "I am waiting for delivery of my bank card", "What is the estimated card arrival time",
    ],
    "cash_withdrawal_charge": [
        "I was charged a fee for a cash withdrawal", "Why did the ATM withdrawal include a fee", "There is an extra cash withdrawal charge",
        "Please explain the fee on my ATM cash withdrawal", "My withdrawal was charged an additional amount",
    ],
    "cash_withdrawal_not_recognised": [
        "I do not recognize this cash withdrawal", "There is an unknown ATM withdrawal on my account", "I did not make this cash withdrawal",
        "Please investigate an unrecognized ATM transaction", "A cash withdrawal appears that is not mine",
    ],
    "pending_transfer": [
        "My bank transfer is still pending", "Why has my transfer not completed", "The transfer status remains pending",
        "Please check the pending bank transfer", "My transfer has not been processed yet",
    ],
    "transfer_fee_charged": [
        "I was charged a fee for a bank transfer", "Why is there an extra fee on my transfer", "The transfer included an unexpected charge",
        "Please explain the bank transfer fee", "There is an additional transfer charge on my account",
    ],
    "card_payment_not_recognised": [
        "I do not recognize this card payment", "There is an unknown card purchase on my account", "I did not make this card transaction",
        "Please investigate an unrecognized card payment", "A card payment appears that is not mine",
    ],
    "declined_cash_withdrawal": [
        "My cash withdrawal was declined", "The ATM refused my withdrawal", "Why was I unable to withdraw cash",
        "The ATM did not allow the requested withdrawal", "My withdrawal attempt was rejected",
    ],
    "wrong_amount_of_cash_received": [
        "The ATM dispensed the wrong amount of cash", "I received less cash than I requested", "The withdrawal amount and cash received do not match",
        "The ATM gave me an incorrect amount", "I requested one amount but received another",
    ],
}

COLLOQUIAL = {
    "card_arrival": [
        "where's my new plastic", "still waiting on that card", "how long till the card shows up", "card delivery taking forever", "when does my card land",
    ],
    "cash_withdrawal_charge": [
        "atm nicked me a fee", "got dinged extra for taking cash out", "why'd the cash machine charge me", "atm fee came out of nowhere", "cash withdrawal cost me extra",
    ],
    "cash_withdrawal_not_recognised": [
        "who took cash from my account", "atm cash out isn't mine", "random cash withdrawal showing", "never made this atm withdrawal", "cash got taken and it wasn't me",
    ],
    "pending_transfer": [
        "transfer is stuck", "money send is still hanging", "why hasn't my transfer gone through", "transfer just says pending", "sent money and it's going nowhere",
    ],
    "transfer_fee_charged": [
        "got hit with a transfer fee", "why'd sending money cost extra", "random fee on my transfer", "transfer charge makes no sense", "they added a fee when i sent money",
    ],
    "card_payment_not_recognised": [
        "this card spend isn't mine", "random card charge showing", "never bought this with my card", "who used my card for this", "card payment popped up that i don't know",
    ],
    "declined_cash_withdrawal": [
        "atm won't give me cash", "cash machine said nope", "why can't i take money out", "atm keeps rejecting me", "withdrawal got blocked",
    ],
    "wrong_amount_of_cash_received": [
        "atm gave me the wrong cash", "cash machine shorted me", "got less money than i asked for", "atm amount was off", "machine spat out the wrong amount",
    ],
}

FILLERS = [
    "today", "this morning", "on my account", "please help", "can you check", "what happened", "i need help", "right now",
]

@dataclass(frozen=True)
class Example:
    text: str
    label: int
    intent: str
    domain: str


def _augment(base: str, rng: random.Random, domain: str) -> str:
    text = base
    if rng.random() < 0.65:
        text = f"{text} {rng.choice(FILLERS)}"
    if rng.random() < 0.35:
        text = f"{rng.choice(['hi', 'hello', 'hey', 'support'])} {text}"
    if domain == "colloquial" and rng.random() < 0.7:
        text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def build_dataset(seed: int = 17, train_per_intent: int = 80, eval_per_intent: int = 30) -> dict[str, list[Example]]:
    rng = random.Random(seed)
    out: dict[str, list[Example]] = {"formal_train": [], "formal_eval": [], "colloquial_train": [], "colloquial_eval": []}
    for label, intent in enumerate(INTENTS):
        for domain, bank in [("formal", FORMAL), ("colloquial", COLLOQUIAL)]:
            seen: set[str] = set()
            pool: list[str] = []
            target = train_per_intent + eval_per_intent
            while len(pool) < target:
                text = _augment(rng.choice(bank[intent]), rng, domain)
                suffix = rng.choice(["", " please", " thanks", " urgently", " again", " now"])
                candidate = (text + suffix).strip()
                if candidate not in seen:
                    seen.add(candidate)
                    pool.append(candidate)
            rng.shuffle(pool)
            train_key = f"{domain}_train"
            eval_key = f"{domain}_eval"
            out[train_key].extend(Example(x, label, intent, domain) for x in pool[:train_per_intent])
            out[eval_key].extend(Example(x, label, intent, domain) for x in pool[train_per_intent:])
    for values in out.values():
        rng.shuffle(values)
    return out


def build_vocab(splits: dict[str, list[Example]]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for split in splits.values():
        for ex in split:
            for tok in re.findall(r"[a-z0-9']+", ex.text.lower()):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int = 24) -> list[int]:
    ids = [vocab.get(tok, 1) for tok in re.findall(r"[a-z0-9']+", text.lower())][:max_len]
    return ids + [0] * (max_len - len(ids))
