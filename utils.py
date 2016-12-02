import json
import os.path


def get_face_data(img_path, img_json_path, from_store=True):
    if not os.path.isfile(img_path):
        return

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
    import httplib, urllib, base64

    subscription_key = 'b0e74930eb5d4a5d984dc1221df41f7e'  # Replace with a valid Subscription Key here.

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

    try:
        conn = httplib.HTTPSConnection('api.projectoxford.ai')
        conn.request("POST", "/face/v1.0/detect?%s" % params, str(body), headers=headers)
        response = conn.getresponse()
        result_text = response.read()
        conn.close()
        return result_text
    except Exception as e:
        print "could not retrieve face api results"
        return None
