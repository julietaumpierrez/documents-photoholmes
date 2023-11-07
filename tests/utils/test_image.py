from photoholmes.utils.image import read_jpeg_data


# ======== DCT channels LOAD ========================================================
def test_read_one_dct_channels():
    dct_channels, _ = read_jpeg_data(
        "tests/images/test_jpeg_image.jpeg", num_dct_channels=1
    )
    assert (
        dct_channels.shape[0] == 1
    ), f"Expected 1 dct_channels, got {dct_channels.shape[0]}"


def test_read_two_dct_channels():
    dct_channels, _ = read_jpeg_data(
        "tests/images/test_jpeg_image.jpeg", num_dct_channels=2
    )
    assert (
        dct_channels.shape[0] == 2
    ), f"Expected 3 dct_channels, got {dct_channels.shape[0]}"


def test_read_three_dct_channels():
    dct_channels, _ = read_jpeg_data(
        "tests/images/test_jpeg_image.jpeg", num_dct_channels=3
    )
    assert (
        dct_channels.shape[0] == 3
    ), f"Expected 3 dct_channels, got {dct_channels.shape[0]}"


# ======== QUANT TABLE LOAD ========================================================
def test_read_one_quant_tables():
    _, qtables = read_jpeg_data(
        "tests/images/test_jpeg_image.jpeg", num_dct_channels=1, all_quant_tables=False
    )
    assert len(qtables) == 1, f"Expected 1 qtable, got {len(qtables)}"


def test_read_all_quant_tables():
    _, qtables = read_jpeg_data(
        "tests/images/test_jpeg_image.jpeg", all_quant_tables=True
    )
    assert len(qtables) == 2, f"Expected 2 qtable, got {len(qtables)}"
