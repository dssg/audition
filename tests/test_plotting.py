from audition.plotting import generate_plot_lines, category_colordict
import matplotlib.lines as mlines


def test_generate_plot_lines():
    colordict = {
        'cat1': '#001122',
        'cat2': '#112233',
        'cat3': '#223344',
    }
    label_fcn = lambda x: 'Cat {}'.format(x)

    plot_lines = generate_plot_lines(colordict, label_fcn)
    assert len(plot_lines) == 3
    for line in plot_lines:
        assert type(line) == mlines.Line2D
        assert 'Cat ' in line._label


def test_category_colordict():
    cmap_name = 'Vega10'
    categories = ['Cat1', 'Cat2', 'Cat3', 'Cat4']
    colordict = category_colordict(cmap_name, categories)
    assert len(colordict.keys()) == 4
