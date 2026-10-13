import unittest

from DyneTrion import eval_actor_pipeline
from src.model.actor import LoRALinear, MetadynamicsActor


def _make_actor():
    return MetadynamicsActor(
        c_s=32,
        c_z=16,
        c_hidden=32,
        no_heads=4,
        no_qk_points=2,
        no_v_points=2,
        num_ipa_blocks=1,
        max_rope_len=32,
        coordinate_scaling=0.1,
    )


class EvalActorPipelineLoRATests(unittest.TestCase):
    def test_auto_prepare_actor_for_lora_state_dict(self):
        base = _make_actor()
        lora_actor = _make_actor()
        lora_actor.load_state_dict(base.state_dict(), strict=True)
        lora_actor.inject_lora(rank=2, alpha=12.0)
        lora_state = lora_actor.state_dict()

        actor = _make_actor()
        meta = eval_actor_pipeline._maybe_prepare_actor_for_state_dict(
            actor, lora_state, {"lora_alpha": 12.0}
        )

        self.assertIsNotNone(meta)
        self.assertEqual(meta["rank"], 2)
        self.assertAlmostEqual(meta["alpha"], 12.0)
        self.assertIsInstance(actor.ipa_layers[0]["ipa"].linear_q, LoRALinear)

        actor.load_state_dict(lora_state, strict=True)

    def test_plain_state_dict_does_not_inject_lora(self):
        actor = _make_actor()
        plain_state = actor.state_dict()

        fresh = _make_actor()
        meta = eval_actor_pipeline._maybe_prepare_actor_for_state_dict(
            fresh, plain_state, {"lora_alpha": 12.0}
        )

        self.assertIsNone(meta)
        self.assertNotIsInstance(fresh.ipa_layers[0]["ipa"].linear_q, LoRALinear)
        fresh.load_state_dict(plain_state, strict=True)


if __name__ == "__main__":
    unittest.main()
