import yagmail
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('recipient', type=str)
parser.add_argument('subject', type=str)
parser.add_argument('content', type=str)
args = parser.parse_args()

user = 'ohayre.hhlabs.user1@gmail.com'
app_pw = 'fcwgizndsckakhxg'

with yagmail.SMTP(user, app_pw) as yag:
    yag.send(args.recipient, args.subject, args.content, prettify_html=False)

