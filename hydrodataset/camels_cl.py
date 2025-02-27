import json
import warnings

    def cache_forcing_np_json(self):
        """
        Save all basin-forcing data in a numpy array file in the cache directory.

        Because it takes much time to read data from txt files,
        it is a good way to cache data as a numpy file to speed up the reading.
        In addition, we need a document to explain the meaning of all dimensions.
        """
        cache_npy_file = CACHE_DIR.joinpath("camels_cl_forcing.npy")
        json_file = CACHE_DIR.joinpath("camels_cl_forcing.json")
        variables = self.get_relevant_cols()
        basins = self.sites["gauge_id"].values  #todo:
        t_range = ["1995-01-01", "2015-01-01"]
        times = [
            hydro_time.t2str(tmp)
            for tmp in hydro_time.t_range_days(t_range).tolist()
        ]
        data_info = collections.OrderedDict(
            {
                "dim": ["basin", "time", "variable"],
                "basin": basins.tolist(),
                "time": times,
                "variable": variables.tolist(),
            }
        )
        with open(json_file, "w") as FP:
            json.dump(data_info, FP, indent=4)
        data = self.read_relevant_cols(
            gage_id_lst=basins.tolist(),
            t_range=t_range,
            var_lst=variables.tolist(),
        )
        np.save(cache_npy_file, data)

