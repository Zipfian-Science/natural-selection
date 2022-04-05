import socketserver
from typing import Tuple
from urllib.parse import urlparse
from linkedin import linkedin
from socketserver import ThreadingTCPServer, TCPServer
from http.server import SimpleHTTPRequestHandler
from webbrowser import open_new_tab
import json
import requests


class LinkedInWrapper(object):

    def __init__(self, id, secret, port):
        self.id = id
        self.secret = secret

        self.callback_url = 'http://localhost:{0}/code/'.format(port)

        print("CLIENT ID: %s" % self.id)
        print("CLIENT SECRET: %s" % self.secret)
        print("Callback URL: %s" % self.callback_url)

        self.authentication = linkedin.LinkedInAuthentication(
            self.id,
            self.secret,
            self.callback_url,
            permissions=['w_member_social']
        )

        self.application = linkedin.LinkedInApplication(self.authentication)

        print("Please double check that the callback URL has been correctly "
              "added in the developer console ("
              "https://www.linkedin.com/developer/apps/), then open "
              "http://localhost:8080 in your browser\n\n")

class LinkedInShareCustomHandler(SimpleHTTPRequestHandler):

    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer):

        self.LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
        self.LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
        self.redirect_uri = "http://localhost:8080/code/"
        self.org_id = int(os.getenv("LINKEDIN_ORGANISATION_ID", 0))

        base_url = "https://www.linkedin.com/oauth/v2/authorization"
        redirect_uri = "http://localhost:8080/code/"
        # scope = "w_organization_social"
        scope = "w_member_social"

        url = f"{base_url}?response_type=code&client_id={self.LINKEDIN_CLIENT_ID}&state=random&redirect_uri={redirect_uri}&scope={scope}"
        print(url)
        super().__init__(request, client_address, server)


    def json_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def get_access_token(self, authorization_code):
        # Get access token

        url_access_token = "https://www.linkedin.com/oauth/v2/accessToken"
        auth_code = authorization_code

        payload = {
            'grant_type': 'authorization_code',
            'code': auth_code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.LINKEDIN_CLIENT_ID,
            'client_secret': self.LINKEDIN_CLIENT_SECRET
        }

        response = requests.post(url=url_access_token, params=payload)
        response_json = response.json()

        # Extract the access_token from the response_json
        return response_json['access_token']

    def get_person_id(self, access_token):
        url = "https://api.linkedin.com/v2/me"
        header = {
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.get(url=url, headers=header)
        response_json_li_person = response.json()
        return response_json_li_person['id']

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

            img_url = 'https://images.pexels.com/photos/2115217/pexels-photo-2115217.jpeg'
            post_external_link = "https://www.redhat.com/en/topics/api/what-is-a-rest-api"
            post_title = "What is a REST API?"
            post_owner = f'urn:li:person:{self.get_person_id(access_token)}' # f'urn:li:organization:{self.org_id}'
            post_text = f'Learn more about REST APIs in details.  \n#restapi #api'

            payload = {
                "content": {
                    "contentEntities": [
                        {
                            "entityLocation": post_external_link,
                            "thumbnails": [
                                {
                                    "resolvedUrl": img_url
                                }
                            ]
                        }
                    ],
                    "title": post_title
                },
                'distribution': {
                    'linkedInDistributionTarget': {}
                },
                'owner': post_owner,
                'text': {
                    'text': post_text
                }
            }

            response = requests.post(url=url, headers=headers, json=payload)

            print(response.json())


import os

def broadcast_update_announce():

    LINKEDIN_PORT = int(os.getenv("LINKEDIN_PORT"))

    LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")

    base_url = "https://www.linkedin.com/oauth/v2/authorization"
    redirect_uri = "http://localhost:8080/code/"
    # scope = "w_organization_social"
    scope = "w_member_social"
    scope = "w_member_social,r_liteprofile"

    url = f"{base_url}?response_type=code&client_id={LINKEDIN_CLIENT_ID}&state=random&redirect_uri={redirect_uri}&scope={scope}"
    print(url)

    httpd = TCPServer(('localhost', LINKEDIN_PORT), LinkedInShareCustomHandler)

    print(('Server started on port:{}'.format(LINKEDIN_PORT)))

    httpd.serve_forever()

    httpd.shutdown()

# broadcast_update_announce()