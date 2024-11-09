from ray.tune import register
register("MyEnv", lambda config: MyEnv(config))
