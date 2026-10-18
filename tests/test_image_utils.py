import pytest
from PIL import Image

from cardgen.image_utils import check_image, fit_image, is_blank, safe_zone_report, save_png_atomic


def make(size=(200, 300), color=(200, 50, 50)):
    img = Image.new("RGB", size, color)
    img.putpixel((0, 0), (0, 0, 0))
    return img


def test_save_and_check_png(tmp_path):
    path = tmp_path / "a" / "card.png"
    save_png_atomic(make(), path, {"cardgen:seed": "5"})
    assert check_image(path, (200, 300)).ok
    with Image.open(path) as img:
        assert img.format == "PNG" and img.text["cardgen:seed"] == "5"
    assert [p.name for p in path.parent.iterdir()] == ["card.png"]


def test_check_detects_problems(tmp_path):
    assert check_image(tmp_path / "nope.png").reason == "missing"
    (tmp_path / "empty.png").write_bytes(b"")
    assert check_image(tmp_path / "empty.png").reason == "empty file"
    (tmp_path / "junk.png").write_bytes(b"not an image at all")
    assert "unreadable" in check_image(tmp_path / "junk.png").reason
    good = tmp_path / "good.png"
    make().save(good)
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(good.read_bytes()[:-40])
    assert not check_image(truncated).ok
    wrong = check_image(good, (512, 768))
    assert not wrong.ok and wrong.reason.startswith("size") and wrong.size == (200, 300)
    jpg = tmp_path / "x.png"
    make().save(jpg, format="JPEG")
    assert "not a PNG" in check_image(jpg).reason


def test_is_blank():
    assert is_blank(Image.new("RGB", (10, 10)))
    assert not is_blank(make())


@pytest.mark.parametrize("fit", ["crop", "pad"])
def test_fit_same_aspect_is_plain_resize(fit):
    out = fit_image(make((512, 768)), 1024, 1536, fit)
    assert out.size == (1024, 1536)


def test_fit_crop_and_pad_do_not_distort():
    square = Image.new("RGB", (100, 100), (255, 0, 0))
    cropped = fit_image(square, 200, 300, "crop")
    assert cropped.size == (200, 300)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)  # filled, no padding
    padded = fit_image(square, 200, 300, "pad", "#0000ff")
    assert padded.size == (200, 300)
    assert padded.getpixel((100, 5)) == (0, 0, 255)  # pad band on top
    assert padded.getpixel((100, 150)) == (255, 0, 0)
    with pytest.raises(ValueError):
        fit_image(square, 200, 300, "stretch")


def test_safe_zone_report_flags_busy_banner_area():
    img = Image.new("RGB", (200, 300), (128, 128, 128))
    for x in range(0, 200, 4):  # stripes only in the top 12%
        for y in range(0, 30):
            img.putpixel((x, y), (0, 0, 0))
    report = safe_zone_report(img, 0.12, 0.15)
    assert report["top_ratio"] > 1.0
    assert report["bottom_edge_density"] == 0
