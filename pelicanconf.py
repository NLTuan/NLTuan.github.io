AUTHOR = 'Le Tuan Huy Nguyen'
SITENAME = "tony's blog"
SITEURL = "https://nltuan.github.io"

PATH = "content"

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'English'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
    ("Pelican", "https://getpelican.com/"),
)

# Social widget
SOCIAL = (
    ("You can add links in your config file", "#"),
    ("Another social link", "#"),
)

DEFAULT_PAGINATION = 10

# Show excerpts on front page instead of full articles
DEFAULT_ORPHANS = 0
DEFAULT_PAGINATION = 5
SUMMARY_MAX_LENGTH = 50

# Enable summary support
DISPLAY_PAGES_ON_MENU = True
MENUITEMS = (
    ('Archives', '/archives.html'),
    ('Categories', '/categories.html'),
    ('Tags', '/tags.html'),
)

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

THEME = "themes/elegant"
