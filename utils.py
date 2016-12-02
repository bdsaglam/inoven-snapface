import json
import os


def get_face_data(img_path, img_json_path, from_store=True):
    result_text = ''
    if from_store:
        if os.path.isfile(img_json_path):
            with open(img_json_path, 'r') as fp:
                result_text = fp.read()

    if not result_text:
        result_text = faceapi_request_file(img_path)
        with open(img_json_path, 'w') as fp:
            fp.write(result_text)

    result_data = json.loads(result_text)
    return result_data


def faceapi_request_file(img_path):
    import httplib, urllib

    subscription_key = os.environ.get('MS_FACE_API_KEY')
    if subscription_key is None:
        raise EnvironmentError("Define a valid subscription key as environment variable MS_FACE_API_KEY")

    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.urlencode({
        # Request parameters
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'true',
        'returnFaceAttributes': 'gender,headPose,smile,facialHair,glasses',
    })

    with open(img_path, 'rb') as f:
        body = f.read()

    conn = httplib.HTTPSConnection('api.projectoxford.ai')
    conn.request("POST", "/face/v1.0/detect?%s" % params, str(body), headers=headers)
    response = conn.getresponse()
    result_text = response.read()
    conn.close()
    return result_text
