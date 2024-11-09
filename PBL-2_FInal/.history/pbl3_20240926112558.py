from ray.tune import register

tune.register("MyEnv", lambda config: MyEnv(config))
