"""Rejeita seleção inválida antes de chamar a inferência e mantém a busca completa."""
import json
import pytest
from fastapi.testclient import TestClient
from test_api import CONFIG, FakePipeline, send
from privacy_service.main import create_app

@pytest.mark.parametrize('value',['null','[]','{}','bad json',json.dumps({'mode':'selected_categories','category_ids':[]}),
    json.dumps({'mode':'selected_categories','category_ids':['security','security']}),
    json.dumps({'mode':'selected_categories','category_ids':['Other']}),
    json.dumps({'mode':'all_categories','category_ids':['security']})])
def test_invalid_selection_never_calls_model(value):
    calls=[]
    class Pipeline(FakePipeline):
        def analyze(self,*args):
            calls.append(args)
            return super().analyze(*args)
    with TestClient(create_app(CONFIG,lambda _:Pipeline())) as client:
        result=send(client,data={'language':'pt','document_scope':'full_document','assessment_scope':value})
        assert result.status_code==400 and calls==[]
        assert send(client).status_code==200

def test_selection_transmitted_in_canonical_order_and_default_resolved():
    with TestClient(create_app(CONFIG,lambda _:FakePipeline())) as client:
        value={'mode':'selected_categories','category_ids':['security','retention']}
        result=send(client,data={'language':'pt','document_scope':'excerpt','assessment_scope':json.dumps(value)})
        assert result.status_code==200
        assert result.json()['assessment_scope']=={'mode':'selected_categories','category_ids':['retention','security']}
        assert result.json()['scope']=='excerpt'
        assert len(send(client).json()['assessment_scope']['category_ids'])==9
