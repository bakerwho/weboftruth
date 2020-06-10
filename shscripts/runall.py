import weboftruth as wot

for ts in [100, 80, 50]:
    tr_fn, val_fn, test_fn = wot.utils.get_file_names(ts)
    tr_df, val_df, test_df = wot.utils.read_data(tr_fn, val_fn, test_fn,
                                wot.svo_paths[ts])
    tr_kg, val_kg, test_kg = (wot.utils.df_to_kg(df.drop(['true_positive'])
                                ) for df in (tr_df, val_df, test_df))
    for model_type in ['DistMult', 'HolE', 'TransE']:
        if model_type == 'TransE':
            print(f"running {model_type} on config {ts}")
            mod = CustomTransModel(tr_kg, model_type=model_type,
                                        ts=ts)
        else:
            print(f"running {model_type} on config {ts}")
            mod = CustomBilinearModel(tr_kg, model_type=model_type,
                                            ts=ts)
        mod.set_sampler(samplerClass=BernoulliNegativeSampler, kg=tr_kg)
        mod.set_optimizer(optClass=Adam)
        mod.set_loss(lossClass=MarginLoss, margin=0.5)
        if cuda.is_available():
            print("Using cuda.")
            cuda.empty_cache()
            mod.model.cuda()
            mod.loss_fn.cuda()
        mod.train_model(2500, val_kg)
        mod.validate(test_kg, istest=True)
