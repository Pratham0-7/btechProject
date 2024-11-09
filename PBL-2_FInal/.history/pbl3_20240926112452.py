from ray import tune

tune.register("MyEnv", lambda config: MyEnv(config))
