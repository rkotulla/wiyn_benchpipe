

import json
import logging
import os

class Config(object):

    def __init__(self, config_fn):
        if (not os.path.isfile(config_fn)):
            raise OSError("Selected config file [%s] does not exist" % (config_fn))

        self.logger = logging.getLogger("Config")

        self.load_defaults()

        self.config_fn = config_fn
        self.logger.debug("Updating config from file %s" % (self.config_fn))
        with open(self.config_fn, "r") as f:
            conf = json.load(f)
            self.update_config(conf)

    def load_defaults(self):
        self.logger.debug("Loading default values")
        self.config = \
            {
                'raw_directory': '.',
                'cals_directory': '.',
                'output_directory': '.',
                'bias': [],
                'flat': [],
                'comp': [],
                'science': [],
                'linelist': 'scidoc2212.fits',
                'setup': {},
                'output': {
                    'min_wl': -1,
                    'max_wl': -1,
                    'dispersion': 0,
                },
                'sky': {
                    'mode': 'default',
                    'fibers': 'minflux',
                    'wlrange': [6700, 6750]
                    'wlrange': [6700, 6750],
                },
                'plots': 'no',
                'readnoise': 5.0,
                'gain': 1.0,
                'cosmics': {
                    'clean': 'no',
                    'sigfrac': 0.5,
                    'sigclip': 1.5,
                    'niter': 3,
                }
            }
        pass

    def update_config(self, conf, *nested):
        # print(type(nested), *nested)
        nest = list(nested)
        subnest = self.config
        for n in nest:
            if (n not in subnest):
                subnest[n] = {}
            subnest = subnest[n]


        for key in conf:
            # print(key)
            value = conf[key]
            if (type(value) in [str ,list ,int ,float]):
                subnest[key] = value
            elif (type(value) == dict):
                # print("Adding subconfig: %s" % (key))
                _nest = list(nest)
                _nest.append(key)
                self.update_config(value, *_nest)

        #
        pass

    def get(self, *opts, fallback=None):
        subnest = self.config
        nest = [opt for opt in list(opts) if opt is not None]
        found = None

        for n in nest:
            # print("checking ", n)
            if (n not in subnest):
                # print("not found")
                break
            elif (type(subnest[n]) != dict):
                # print("found not dictionary")
                found = subnest[n]
                break
            # print("found nested dictionary")
            subnest = subnest[n]

        if (found is None):
            # nothing found yet
            if (len(nest) > 1):
                return self.get(*nest[1:], fallback=fallback)
            else:
                return fallback

        return found
        # subnest = subnest[n]

        pass

    def set(self, *nested):
        _nested = list(nested)
        nest = _nested[:-2]
        key = _nested[-2]
        value = _nested[-1]

        self.logger.debug("Setting %s to %s" % ("->".join(_nested[:-1]), str(value)))

        subnest = self.config
        for n in nest:
            if (n not in subnest):
                subnest[n] = {}
            subnest = subnest[n]

        subnest[key] = value

    def report(self, *opts):
        value = self.get(*opts)
        self.logger.info("Config for %s: " % (" -> ".join(list(opts))), value)

    def write(self, filename=None):
        txt = json.dumps(self.config, indent=5)
        if (filename is not None):
            with open(filename, "w") as f:
                f.write(txt)
        return txt

    def create_output_directories(self):
        # get list of all output directories
        dir_list = [self.get("cals_directory"),
                    self.get("output_directory")]
        targets = self.get("science")
        # print(targets)
        for target in targets:
            _d = self.get(target, "output_directory")
            # print("%s --> %s" % (target, _d))
            dir_list.append(_d)
        dir_list = [d for d in dir_list if d is not None]
        # print(dir_list)

        self.logger.info("List of all output directories:\n%s" % ("\n ** ".join(dir_list)))

        # check if directory exists, and if not create it
        for dir_name in set(dir_list):
            if (dir_name is None):
                continue
            elif (not os.path.isdir(dir_name)):
                self.logger.info("Creating directory: %s" % (dir_name))
                os.makedirs(dir_name)