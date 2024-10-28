import math
import random
import plotly.express as px

from aim import Run, Distribution, Image, Figure

if __name__ == '__main__':
    run = Run(experiment='test-4')
    run['hparams'] = {
        'hyper_a': 123,
        'lr': 5e-5
    }
    for iteration in range(0, 1000):
        run.track(iteration + random.randint(-50, 50), 'stochastic', iteration)
        run.track(iteration * 0.05, 'metric_1', iteration)
        run.track(math.log(10 + iteration), 'metric_2', iteration)
        run.track(Distribution(
            samples=[random.randrange(0, 10000) for _ in range(1000)],
            bin_count=250
        ), 'my_distribution', iteration)
        run.log_error('error 123')
        print(f'lol {iteration}')
        if iteration % 100 == 0:
            if iteration % 200:
                run.track(Image('/home/me/downloads/Cat.jpg', caption='my cat'), 'generated_image', step=iteration)
            else:
                run.track(Image('/home/me/downloads/images (3).jpeg', caption='my cat'), 'generated_image', step=iteration)

    df = px.data.gapminder().query("year == 2007")
    fig = px.treemap(df, path=[px.Constant('world'), 'continent', 'country'], values='pop',
                     color='lifeExp', hover_data=['iso_alpha'])
    run.track(Figure(fig), 'life_exp', step=1000)
    run.close()
