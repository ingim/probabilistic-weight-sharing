import torch
import model
import umap
import seaborn as sns


def main():
    l1_dim = 128
    l2_dim = 32

    mode = 'pps'
    num_f = 5
    num_epoch = 1000
    use_bias = False
    var_size = [2048, 64, 1]

    input_size = 32 * 96  # 3072

    baseline = model.Simple(input_size, 1, 10, mode=mode, num_f=num_f, l1_dim=l1_dim, l2_dim=l2_dim, var_size=var_size,
                            bias=use_bias,
                            temperature=0.1).cuda()

    baseline.load_state_dict(torch.load('./pps3.pt'))

    l1_dist_probs = baseline.l1.dist_probs.data
    l2_dist_probs = baseline.l2.dist_probs.data
    l3_dist_probs = baseline.l3.dist_probs.data

    l1_weight = baseline.l1.weight.data
    l2_weight = baseline.l2.weight.data
    l3_weight = baseline.l3.weight.data

    # create a categorical distribution parameterized by l1_dist
    l1_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(0.01, logits=l1_dist_probs)
    l2_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(0.01, logits=l2_dist_probs)
    l3_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(0.01, logits=l3_dist_probs)

    num_samples = 100

    label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 10, 1).repeat(num_samples, 1, 1).view(-1)

    def fig2():
        # (N, T, num_vars, cat)
        onehot = l3_dist.sample([num_samples])

        # Expand and reshape one-hot vectors
        # (N, T, num_vars, C) -> (N, T, num_vars, var_size, C)
        onehot_exp = onehot.unsqueeze(-2).expand(-1, -1, -1, var_size[2], -1)

        # (N, T, num_vars, var_size, C) * (1, 1, num_vars, var_size, C) -> (N, T, num_vars, var_size, C)
        sampled_weight = onehot_exp * l3_weight.unsqueeze(0).unsqueeze(0)

        # (N, T, num_vars, var_size, C) -> (N, num_features)
        sampled_weight = torch.sum(sampled_weight, dim=-1).view(num_samples * 10, -1)

        samples = sampled_weight.to('cpu')

        # (1000, 2)
        fit = umap.UMAP(
            densmap=True,
            n_neighbors=5,
            min_dist=0.8,
            metric='euclidean',
            random_state=42, low_memory=False).fit(samples)

        embedding = fit.fit_transform(samples)

        print(samples.shape)
        print(label.shape)

        # save the figure
        sns_plot = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label, s=1,
                                   palette=sns.color_palette("Paired", 10))

        # legend as a right-side bar

        # remove x and y axis
        sns_plot.axes.get_xaxis().set_visible(False)
        sns_plot.axes.get_yaxis().set_visible(False)

        # make the dots smaller

        sns_plot.figure.savefig("output.png")

    fig2()

    def fig1():
        # (100, T, num_vars * cat) is the new feature vector
        samples = l3_dist.sample([num_samples]).view(num_samples, 10, -1).view(num_samples * 10, -1)

        # (1000, 2)
        fit = umap.UMAP(
            densmap=True,
            n_neighbors=5,
            min_dist=0.8,
            metric='euclidean',
            random_state=42, low_memory=False).fit(samples)

        embedding = fit.fit_transform(samples)

        print(samples.shape)
        print(label.shape)

        # save the figure
        sns_plot = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=label[:, 0], s=1,
                                   palette=sns.color_palette("Paired", 10))

        # legend as a right-side bar

        # remove x and y axis
        sns_plot.axes.get_xaxis().set_visible(False)
        sns_plot.axes.get_yaxis().set_visible(False)

        # make the dots smaller

        sns_plot.figure.savefig("output.png")


if __name__ == "__main__":
    main()
