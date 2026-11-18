import torch

from ultralytics_ext.adversarial_feature_kd import AdversarialFeatureKD


def test_discriminator_is_independent_and_student_receives_gradient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kd = AdversarialFeatureKD({"adv_p5": (256, False)}, device)
    student = {"adv_p5": torch.randn(2, 256, 20, 20, device=device, requires_grad=True)}
    teacher = {"adv_p5": torch.randn(2, 256, 20, 20, device=device, requires_grad=True)}
    stats = kd.discriminator_step(student, teacher); loss = kd.generator_losses(student)["adv_p5"]; loss.backward()
    assert torch.isfinite(loss) and stats["d_loss"] > 0; assert student["adv_p5"].grad is not None; assert teacher["adv_p5"].grad is None
