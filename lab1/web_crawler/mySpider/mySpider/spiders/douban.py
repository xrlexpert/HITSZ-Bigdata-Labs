import scrapy
from scrapy import Selector
from ..items import Item

class DoubanSpider(scrapy.Spider):
    name = "douban"
    allowed_domains = ["movie.douban.com"]
    start_urls = [f'https://movie.douban.com/top250?start={i}&filter=' for i in range(0, 250, 25)]

    def parse(self, response):
        print(response.url)
        sel = Selector(response)
        movies_url = response.css("#content > div > div.article > ol > li > div > div.info > div.hd > a::attr(href)").getall()
        titles = response.css("#content > div > div.article > ol > li:nth-child(1) > div > div.info > div.hd > a > span::text").getall()
        for movie_url  in movies_url:
            print(movie_url)
            yield scrapy.Request(url=movie_url, callback=self.parse_comment_page)
    def parse_comment_page(self, response):
        comment_pages_url = [response.urljoin(f"comments?start={i}&limit=20&status=P&sort=new_score") for i in range(0, 280, 20)]
        for comment_page_url in comment_pages_url:
            yield scrapy.Request(url=comment_page_url, callback=self.parse_comment)

    def parse_comment(self, response):
        comments = response.css("#comments > div > div.comment > p > span::text").getall()
        print(len(comments))
        for comment in comments:
            item = Item()
            item["text"] = comment
            yield item