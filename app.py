import os
import csv
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

class SatelliteViewHandler(tornado.web.RequestHandler):
    def cluster(self, spec):
        if spec == 'setosa':
            return 0
        elif spec == 'versicolor':
            return 1
        elif spec == 'virginica':
            return 2

    def get(self):
        with open(os.path.join(os.getcwd(), "static", "data.tsv"),'r') as df:
            reader = csv.DictReader(df, delimiter = '\t')
            data = []
            for row in reader:
                data.append([float(row['sepalLength']), float(row['sepalWidth']), self.cluster(row['species'])])
            print(data.shape)
        self.render('satelliteview.html', data=data)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/satview", SatelliteViewHandler),
    ],
    template_path=os.path.join(os.getcwd(), "templates"),
    static_path=os.path.join(os.getcwd(), "static"),
    )

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
