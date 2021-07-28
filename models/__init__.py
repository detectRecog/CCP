from models.BranchedERFNet import *


def get_model(name, model_opts):
    if name == "branched_erfnet_up4":
        model = BranchedERFNetUp4(**model_opts)
        return model
    elif name == "pointtrack-v8-rec":
        from models.Lightning_PointTrackV8Rec import PointTrackLightning
        model = PointTrackLightning(**model_opts)
        return model
    elif name == "ccpnet":
        from models.Lightning_CCPNet import PointTrackLightning
        model = PointTrackLightning(**model_opts)
        return model
    elif name == "ccpnet_track":
        from models.Lightning_CCPNet_track import PointTrackLightning
        model = PointTrackLightning(**model_opts)
        return model
    elif name == "tracker_offset_emb_randla":
        from models.modelsPami import TrackerOffsetEmbRandLA
        model = TrackerOffsetEmbRandLA(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))