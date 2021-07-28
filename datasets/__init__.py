
def get_dataset(name, dataset_opts):
    if name == "mots_cars_CAP": # Copy-and-Paste
        from datasets.MOTSCopyAndPaste import MOTSCopyAndPasteCars
        return MOTSCopyAndPasteCars(**dataset_opts)
    elif name == "mots_cars_CAP_v3": # Copy-and-Paste
        from datasets.MOTSCopyAndPaste import MOTSCopyAndPasteCarsV3
        return MOTSCopyAndPasteCarsV3(**dataset_opts)
    elif name == "mots_cars_CAP_v2": # Copy-and-Paste
        from datasets.MOTSCopyAndPaste import MOTSCopyAndPasteCarsV2
        return MOTSCopyAndPasteCarsV2(**dataset_opts)
    elif name == "person_track_val_offset":
        from datasets.KittiMOTSDataset import PersonTrackValOffset
        return PersonTrackValOffset(**dataset_opts)
    elif name == "mots_track_val_env_offset":
        from datasets.KittiMOTSDataset import MOTSTrackCarsValOffset
        return MOTSTrackCarsValOffset(**dataset_opts)
    elif name == "mots_cars_CCAP": # Continuous Copy-and-Paste
        from datasets.MOTSContinousCopyAndPaste import MOTSContinousCopyAndPasteCars
        return MOTSContinousCopyAndPasteCars(**dataset_opts)
    elif name == "mots_cars_CCAP_v2": # Continuous Copy-and-Paste
        from datasets.MOTSContinousCopyAndPaste import MOTSContinousCopyAndPasteCarsV2
        return MOTSContinousCopyAndPasteCarsV2(**dataset_opts)
    elif name == "mots_cars_CCAP_v3": # Continuous Copy-and-Paste
        from datasets.MOTSContinousCopyAndPaste import MOTSContinousCopyAndPasteCarsV3
        return MOTSContinousCopyAndPasteCarsV3(**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))