import yaml


def load_config(config_path, logger):
    with open(config_path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
