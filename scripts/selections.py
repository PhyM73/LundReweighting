def tau21_selections(jets, tau21_cut=0.4):
    tau1 = jets[:, 4]
    tau2 = jets[:, 5]
    # Correct definition of tau21 is tau2 / tau1
    eps = 1e-6
    mask = (tau2 / (tau1 + eps)) < tau21_cut
    return mask