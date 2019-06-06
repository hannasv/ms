class ProcessSateliteData:
    """
    Class for processing sataelite data.

    It needs to be:
        1. Cropped to the correct regions.
        2. Replace "no cloud over ocean and land" with "no cloud".
        3. Regridd and calculate cloud fractions
        4. Merge timesteps together. Preferably get it into one file.
            * consider seasonal files
            * consider train test split files.
            
        OBS! Here you can considerer the
        train test split when merging the timesteps.

        TODO: train test split should maybee be inported from a config.
        And should have one script to run which updates all files with a new train test split.


    """

      def __init__():
        pass

        def regrid_tcc():
            """



            Regrid the cloud mask to total cloud cover, which has the same spatial
             resolution as era interim data.z<
            """
            pass
