from hydrodataset import CamelsCcam

def test_read_forcing():
    camelsccam = CamelsCcam()
    gage_ids = camelsccam.read_object_ids()
    print(gage_ids)
    forcings = camelsccam.read_relevant_cols(
        gage_ids[:5],
        ["1990-01-01", "2021-04-01"],
        var_lst=[
                "pre",
                "evp",
                "gst_mean",
                "prs_mean",
                "tem_mean",
                "rhu",
                "win_mean",
                "gst_min",
                "prs_min",
                "tem_min",
                "gst_max",
                "prs_max",
                "tem_max",
                "ssd",
                "win_max",
        ]
    )
