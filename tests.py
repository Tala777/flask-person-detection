import os
import main
import unittest
import tempfile


class FlaskBookshelfTests(unittest.TestCase):

    def setUp(self):
        # creates a test client
        main.app.config['TESTING'] = True
        main.app.config['WTF_CSRF_ENABLED'] = False
        main.app.config['DEBUG'] = False
        main.app.secret_key = 'my unobvious secret key'
        self.app = main.app.test_client()
        # propagate the exceptions to the test client
        self.app.testing = True

    def tearDown(self):
        pass

    def test_upload_file__get__status_code(self):
        # sends HTTP GET request to the application
        # on the specified path
        result = self.app.get('/')

        # assert the status code of the response
        self.assertEqual(result.status_code, 200)

    def test_upload_file__post__image_file_processed(self):
        # sends HTTP GET request to the application
        # on the specified path
        data = dict(
            file='@{}'.format(os.path.abspath('test_data/images/test_image_1.jpg')),
            file_type="image",
            threshold_person="0.5",
            threshold_face="0.5"
        )
        response = self.app.post(
            "/",
            data=data,
            content_type='multipart/form-data',
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
