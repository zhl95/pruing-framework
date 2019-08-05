
import torch
import torch.nn as nn
from config_ import cfg



class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.called = 0

    def init_mask(self):
        if cfg.MODE == 'gm' or cfg.MODE == 'merge':
            print(cfg.MODE, 'whitin model init mask')
            for m in self.modules():
                # print("init mask for", m)
                if isinstance(m, nn.Conv2d):
                    # print(self.called, "th to mask layer: ", m)
                    # print(m)
                    try:
                        m.do_prune()
                    except AttributeError:
                        # print("layer DataParallel wrapped")
                        m.module.do_prune()
        elif cfg.MODE == 'regu':
            pass
            
        
        elif cfg.MODE == "sup":
            for m in self.modules():
                # print("init mask for", m)
                if isinstance(m, nn.Conv2d):
                    # print(self.called, "th to mask layer: ", m)
                    # print(m)
                    try:
                        m.init_mask()
                    except AttributeError:
                        print("layer DataParallel wrapped")
                        m.module.init_mask()
        else:
            pass


    def to_mask(self):
        if cfg.MODE == 'gm' or cfg.MODE == 'merge':
            self.called += 1
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(self.called, "th to mask layer: ", m)
                    # print(m)
                    try:
                        m.to_mask()
                    except AttributeError:
                        # print("layer DataParallel wrapped")
                        m.module.to_mask()
        elif cfg.MODE == 'regu':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(self.called, "th to mask layer: ", m)
                    # print(m)
                    try:
                        m.to_regu()
                    except AttributeError:
                        # print("layer DataParallel wrapped")
                        m.module.to_regu() 
        elif cfg.MODE == 'sup':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(self.called, "th to mask layer: ", m)
                    # print(m)
                    try:
                        m.to_sup()
                    except AttributeError:
                        print("layer DataParallel wrapped")
                        m.module.to_sup() 
        else:
            pass

    def get_simi(self):
        simi = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print(self.called, "th to mask layer: ", m)
                # print(m)
                try:
                    simi += m.cos_loss
                except AttributeError:
                    print("layer DataParallel wrapped")
                    simi += m.module.cos_loss
        return simi

    def shrink(self):
        if cfg.MODE == 'regu':
            for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        # print(self.called, "th to mask layer: ", m)
                        # print(m)
                        try:
                            m.to_mask()
                        except AttributeError:
                            # print("layer DataParallel wrapped")
                            m.module.to_mask()
        elif cfg.MODE == 'gm' or cfg.MODE == 'merge':          
            pass 
        else:
            pass

    def mask_grad(self):
        if cfg.MODE == 'gm' or cfg.MODE == 'merge':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print("to mask grad layer: ", m)
                    # print(m)
                    try:
                        m.mask_grad()
                    except AttributeError:
                        # print("DataParallel wrapped")
                        m.module.mask_grad()
        elif cfg.MODE == 'regu':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print("to mask grad layer: ", m)
                    # print(m)
                    try:
                        # m.check_grad()
                        m.regu_grad()
                    except AttributeError:
                        # print("DataParallel wrapped")
                        m.module.regu_grad()
        else:
            pass

    def cos_loss(self):
        cos_loss_ = 0
        if cfg.MODE == 'regu':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    assert not m.cos_loss is None
                    try:
                        cos_loss_ += m.cos_loss
                    except AttributeError:
                        # print("DataParallel wrapped")
                        cos_loss_ += m.module.cos_loss
        elif cfg.MODE == 'gm' or cfg.MODE == 'merge':
            pass 
        else:
            pass
        return cos_loss_

    
