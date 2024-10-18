import scrapy


class ItcastSpider(scrapy.Spider):
    name = "itcast"
    allowed_domains = ["itcast.cn"]
    start_urls = ["http://www.itcast.cn/channel/teacher.shtml",]

    def parse(self, response):
        filename = "teacher.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.body.decode('utf-8'))
        pass
