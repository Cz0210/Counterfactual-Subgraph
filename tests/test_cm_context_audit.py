from types import SimpleNamespace
from src.baselines.cm_crem_audit import original_filter_context

def test_original_context_includes_rejected_companions_and_order():
    rows=[{'candidate_id':str(i)} for i in range(70)]
    class Oracle:
        oracle=SimpleNamespace(default_batch_size=32)
        def predict_rows(self,rows,*,split):return rows
        def filter_generated(self,parent,generated):
            self.predict_rows([parent],split='train')
            self.predict_rows(rows,split='train_generated')
    o=Oracle(); original=o.predict_rows
    def read(name):return {'parents':[{'parent_id':'p'}]} if name=='attribution.json' else {}
    context,pos,origin=original_filter_context(o,read,{'candidate_id':'40','origins':[{'parent_id':'p'}]})
    assert pos==8 and [r['candidate_id'] for r in context]==[str(i) for i in range(32,64)]
    assert o.predict_rows==original
