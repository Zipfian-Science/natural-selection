import socketserver
from typing import Tuple
from urllib.parse import urlparse
from socketserver import ThreadingTCPServer
from http.server import SimpleHTTPRequestHandler
import threading
import requests
import os
import webbrowser

class LinkedInShareCustomHandler(SimpleHTTPRequestHandler):

    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer):

        self.LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
        self.LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
        self.redirect_uri = "http://localhost:8080/code/"
        self.org_id = int(os.getenv("LINKEDIN_ORGANISATION_ID", 0))

        self.scope = os.getenv("LINKEDIN_SCOPE", "w_organization_social")

        super().__init__(request, client_address, server)


    def json_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def get_access_token(self, authorization_code):

        url_access_token = "https://www.linkedin.com/oauth/v2/accessToken"

        payload = {
            'grant_type': 'authorization_code',
            'code': authorization_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.LINKEDIN_CLIENT_ID,
            'client_secret': self.LINKEDIN_CLIENT_SECRET
        }

        response = requests.post(url=url_access_token, params=payload)
        response_json = response.json()

        return response_json['access_token']

    def get_person_id(self, access_token):
        url = "https://api.linkedin.com/v2/me"
        header = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(url=url, headers=header)
        response_json_li_person = response.json()
        return response_json_li_person['id']

    def _build_post_details(self):
        changes = ''
        appending = False
        with open('./docs/source/changelog.rst', 'r') as f:
            lines = f.readlines()
            for l in lines:
                if l.startswith('---'):
                    appending = True
                    continue
                if l.startswith('Version') and appending:
                    break
                if appending:
                    changes = f'{changes}{l}'

        changes = f'A new update to the natural-selection package has just been released on PyPI! Changes include:{changes}#genetic #algorithms in #python'

        return {
            'img_url' : 'https://zipfian.science/assets/images/ea_small.png',
            'post_external_link' : "https://docs.zipfian.science/natural-selection/changelog.html",
            'post_title' :  f"New natural-selection updates",
            'post_text' : changes
        }


    def do_GET(self):

        params_to_dict = lambda params: {
            l[0]: l[1] for l in [j.split('=')
                                 for j in urlparse(params).query.split('&')]
        }

        parsedurl = urlparse(self.path)

        if parsedurl.path in ['/code/', '/code']:
            self.json_headers()

            params_dict = params_to_dict(self.path)

            authorization_code = params_dict.get('code')
            access_token = self.get_access_token(authorization_code)

            # Do share

            url = "https://api.linkedin.com/v2/shares"

            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }

            post_details = self._build_post_details()

            if self.scope == 'w_organization_social':
                post_owner = f'urn:li:organization:{self.org_id}'
            else:
                post_owner = f'urn:li:person:{self.get_person_id(access_token)}'

            payload = {
                "content": {
                    "contentEntities": [
                        {
                            "entityLocation": post_details['post_external_link'],
                            "thumbnails": [
                                {
                                    "resolvedUrl": post_details['img_url']
                                }
                            ]
                        }
                    ],
                    "title": post_details['post_title']
                },
                'distribution': {
                    'linkedInDistributionTarget': {}
                },
                'owner': post_owner,
                'text': {
                    'text': post_details['post_text']
                }
            }

            response = requests.post(url=url, headers=headers, json=payload)

            print(response.json())

            server = threading.Thread(target=httpd.shutdown)
            server.daemon = True
            server.start()

LINKEDIN_PORT = int(os.getenv("LINKEDIN_PORT"))
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
base_url = "https://www.linkedin.com/oauth/v2/authorization"
redirect_uri = "http://localhost:8080/code/"

scope = os.getenv("LINKEDIN_SCOPE", "w_organization_social")

url = f"{base_url}?response_type=code&client_id={LINKEDIN_CLIENT_ID}&state=random&redirect_uri={redirect_uri}&scope={scope}"
print(url)

httpd = ThreadingTCPServer(('localhost', LINKEDIN_PORT), LinkedInShareCustomHandler)

def broadcast_update_linkedin():

    with httpd:

        print(('Server started on port:{}'.format(LINKEDIN_PORT)))
        webbrowser.open(url, new=0, autoraise=True)
        httpd.serve_forever()
        webbrowser.open('https://www.linkedin.com/company/zipfian-science/posts/', new=0, autoraise=True)