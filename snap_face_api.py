from flask import Flask
from flask_restful import reqparse, abort, Api, Resource, fields, marshal

app = Flask(__name__)
api = Api(app)

effect_list = [
    {'pk': '1', 'kind': 'glasses', 'name': 'black', 'image_uri': '/static/img/glasses/sg_black.png'},
    {'pk': '2', 'kind': 'glasses', 'name': 'red', 'image_uri': '/static/img/glasses/sg_red.png'},
    {'pk': '3', 'kind': 'glasses', 'name': 'pink', 'image_uri': '/static/img/glasses/sg_pink.png'},
    {'pk': '4', 'kind': 'glasses', 'name': 'aviator_navy', 'image_uri': '/static/img/glasses/sg_aviator_navy.png'},
]

effect_fields = {
    'pk': fields.String,
    'kind': fields.String,
    'name': fields.String,
    'image_uri': fields.String,
    'uri': fields.Url('effect')
}


class EffectAPI(Resource):
    reqparse = reqparse.RequestParser()
    reqparse.add_argument('kind', type=str, location='json')
    reqparse.add_argument('name', type=str, location='json')

    def __init__(self):
        super(EffectAPI, self).__init__()

    def get(self, pk):
        queryset = [effect for effect in effect_list if effect['pk'] == pk]
        if len(queryset) == 0:
            abort(404)
        return {'effect': marshal(queryset[0], effect_fields)}

    def delete(self, pk):
        queryset = [effect for effect in effect_list if effect['pk'] == pk]
        if len(queryset) == 0:
            abort(404)
        effect_list.remove(queryset[0])
        return {'result': True}, 204

    def put(self, pk):
        queryset = [effect for effect in effect_list if effect['pk'] == pk]
        if len(queryset) == 0:
            abort(404)
        effect = queryset[0]
        args = EffectAPI.reqparse.parse_args()
        for k, v in args.items():
            if v is not None:
                effect[k] = v
        return {'effect': marshal(effect, effect_fields)}, 201


class EffectListAPI(Resource):
    def get(self):
        return {'effects': [marshal(effect, effect_fields) for effect in effect_list]}

    def post(self):
        args = EffectAPI.reqparse.parse_args()
        pk_list = [effect.get('pk', '0') for effect in effect_list]
        pk = str(int(max(pk_list) + 1))
        effect = {
            'pk': pk,
            'kind': args['kind'],
            'name': args['name'],
        }
        effect_list.append(effect)
        return {'effect': marshal(effect, effect_fields)}, 201


# Actually setup the Api resource routing here
api.add_resource(EffectListAPI, '/effects/', endpoint='effects')
api.add_resource(EffectAPI, '/effects/<pk>', endpoint='effect')

if __name__ == '__main__':
    app.run(debug=True)
