from Shen import *
from wuli import *
from creature import *
import os,time,subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp

# ==================== PyTorch 模型 ====================

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1=nn.Linear(statnum,30)
        self.fh=nn.ModuleList([nn.Linear(30,30) for i in range(6)])
        self.f2=nn.Linear(30,musclenum*2)

    def forward(self,x):
        x=torch.relu(self.f1(x))
        for layer in self.fh:
            x=x+torch.relu(layer(x))
        x=self.f2(x)
        chunks=x.split(2)
        return torch.cat([torch.softmax(c,dim=-1) for c in chunks])

    def choice(self,ten):
        """接受 Ten 对象（兼容 env.getstat()），返回动作列表"""
        with torch.no_grad():
            probs=self(torch.tensor(ten.data,dtype=torch.float32)).numpy().tolist()
        a=[]
        for i in range(0,len(probs),2):
            a.append(random.choices([0,1],weights=probs[i:i+2])[0])
        return a

class modelv(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1=nn.Linear(statnum,30)
        self.fh=nn.ModuleList([nn.Linear(30,30) for i in range(6)])
        self.f2=nn.Linear(30,1)

    def forward(self,x):
        x=torch.relu(self.f1(x))
        for layer in self.fh:
            x=x+torch.relu(layer(x))
        return self.f2(x)

# ==================== 经验回放 ====================

class memory:
    def __init__(self,maxsize=10):
        self.memo=[]
        self.maxsize=maxsize

    def experience(self,m,times=3,n=500):
        for t in range(times):
            e=env()
            exp=[]
            ar=0
            for i in range(n):
                s_ten=e.getstat()
                s=torch.tensor(s_ten.data,dtype=torch.float32)
                with torch.no_grad():
                    v=m(s).numpy().tolist()
                a=[]
                p=[]
                for j in range(0,len(v),2):
                    probs=v[j:j+2]
                    action=random.choices([0,1],weights=probs)[0]
                    a.append(action)
                    p.append(probs[action])
                e.act(a)
                e.step(0.001)
                st_ten=e.getstat()
                st=torch.tensor(st_ten.data,dtype=torch.float32)
                r=e.reward()
                ar+=r
                exp.append([s,a,r,st,0,p])
                if i==n-1 or e.isend():
                    exp[-1][4]=1
                    break
            self.memo.append(exp)
        if len(self.memo)>self.maxsize:
            self.memo=self.memo[-self.maxsize:]
        return ar/times

# ==================== 训练函数 ====================

def mase_torch(x,y):
    a=((x-y)**2).sum()
    if a.item()<1:
        return a
    else:
        return torch.sqrt(a)

def train(m,mv,opt_m,opt_mv,memo,n=200,times=1,discount=0.99,lamb=0.99,ek=0.5,eps=0.2):
    t0=time.perf_counter()
    ar=memo.experience(m,times,n=n)
    t1=time.perf_counter()
    aloss=0
    alossv=0
    alosse=0
    alratio=0
    count=0

    opt_m.zero_grad()
    opt_mv.zero_grad()

    for exp in memo.memo:
        ad=[]
        gae=0
        for i in range(len(exp)-1,-1,-1):
            with torch.no_grad():
                v=mv(exp[i][0]).item()
                v2=mv(exp[i][3]).item()
            tdd=exp[i][2]+discount*v2*(1-exp[i][4])-v
            gae=tdd+lamb*discount*gae*(1-exp[i][4])
            ad.append(gae)
        ad.reverse()

        for idx,(s,a,r,st,end,p) in enumerate(exp):
            out=m(s)
            pc=torch.cat([out[a[j]+j*2:a[j]+j*2+1] for j in range(len(a))])
            ent=(-out*torch.log(out+1e-8)).sum()
            v_pred=mv(s)
            adv=torch.tensor([ad[idx]],dtype=torch.float32)
            p_tensor=torch.tensor(p,dtype=torch.float32)

            ratio=torch.exp(torch.log(pc+1e-8)-torch.log(p_tensor+1e-8))
            surr=ratio*adv.expand(len(pc))
            surr2=torch.clamp(ratio,1-eps,1+eps)*adv.expand(len(pc))
            loss=-torch.min(surr,surr2).sum()
            losse=-ek*ent

            (loss+losse).backward()

            target=torch.tensor([ad[idx]-v_pred.item()],dtype=torch.float32)
            lossv=mase_torch(v_pred,target)
            lossv.backward()

            aloss+=loss.item()/len(pc)
            alosse+=ent.item()/len(out)
            alratio+=torch.abs(ratio-1).mean().item()
            alossv+=lossv.item()
            count+=1

    for p in m.parameters():
        if p.grad is not None:
            p.grad/=count
    for p in mv.parameters():
        if p.grad is not None:
            p.grad/=count

    opt_m.step()
    opt_mv.step()

    t2=time.perf_counter()
    print(ar,aloss/count,alossv/count,alosse/count,alratio/count,t1-t0,t2-t1)
    return ar,aloss/count,alossv/count,alosse/count

# ==================== 环境（保持不变） ====================

class env(Environment):
    def __init__(self):
        Phy.biao=[]
        Phy.rbiao=[]
        super().__init__([eval(evnname+"()")],
                         g=650,
                         groundhigh=-50,
                         groundk=10000,
                         grounddamp=100,
                         randsigma=0.1,
                         dampk=0.08,
                         friction=650)
        for i in self.creatures[0].muscles:
            i.stride=3
            i.damk=20
        self.r=0
        if len(self.creatures[0].skeletons)!=0:
            self.plp=[self.creatures[0].skeletons[0].p1,self.creatures[0].skeletons[0].p2]
        else:
            self.plp=[self.creatures[0].phys[0],self.creatures[0].phys[1]]
        self.plumb=[(self.plp[1].p[0]-self.plp[0].p[0])/distant(self.plp[0],self.plp[1]),
                    (self.plp[1].p[1]-self.plp[0].p[1])/distant(self.plp[0],self.plp[1])]
        self.ang=0
        self.foot=[i for i in self.creatures[0].phys if i.p[1]<=0]

    def getstat(self):  #box21 leg35
        s=self.creatures[0].getstat(False,pk=0.023,vk=0.028,ak=0.001,mk=0.05)
        return s

    def act(self,a):
        self.creatures[0].actdisp(a)

    def reward(self):
        return self.r

    def show(self,m):
        e=env()
        Phy.tready()
        ar=0
        for i in range(n):
            a=m.choice(e.getstat())
            e.act(a)
            e.step(0.001)
            ar+=e.reward()
            turtle.goto(-800,ground)
            turtle.pendown()
            turtle.goto(800,ground)
            turtle.penup()
            Phy.tplay()
            if e.isend():
                break
            time.sleep(0.01)
        print(ar)

    def step(self,t): # reward
        v=0
        ang=0
        energy=0
        smooth=0
        for i in range(30):
            super().step(t)
            # 速度
            v+=sum([i.v[0] for i in self.creatures[0].phys])/len(self.creatures[0].phys)
            # 姿态
            ang+=((self.plp[1].p[0]-self.plp[0].p[0])*self.plumb[0]\
                +(self.plp[1].p[1]-self.plp[0].p[1])*self.plumb[1])/distant(self.plp[0],self.plp[1])
            # 平滑性
            smooth+=sum([abs(i.a[0])+abs(i.a[1]) for i in self.creatures[0].phys])/len(self.creatures[0].phys)

        # 前进奖励（降低阈值，早期更容易获得）
        self.r=min(v/30,2.0)
        self.ang=ang

        # 姿态惩罚（加强）
        self.r-=max(0,1-ang/30)*0.5

        # 能量效率惩罚（用肌肉长度变化表示）
        energy=sum([abs(m.x-m.originx) for m in self.creatures[0].muscles])/len(self.creatures[0].muscles)
        self.r-=energy*0.0005

        # 平滑性惩罚
        self.r-=smooth*0.0001

        # 摔倒惩罚（加强）
        if self.isend(3):
            self.r-=15

    def test(self,times=10):
        for t in range(times):
            e=env()
            ar=0
            for i in range(n):
                e.act([random.randint(0,1) for i in range(musclenum)])
                e.step(0.001)
                ar+=e.reward()
                p=0
                v=0
                a=0
                m=0
                for i in e.creatures[0].phys:
                    p+=(i.p[0]+i.p[1])/2
                    v+=(i.v[0]+i.v[1])/2
                    a+=(i.axianshi[0]+i.axianshi[1])/2
                for i in e.creatures[0].muscles:
                    m+=distant(i.p1,i.p2)
                p/=len(e.creatures[0].phys)
                v/=len(e.creatures[0].phys)
                a/=len(e.creatures[0].phys)
                m/=len(e.creatures[0].muscles)
                print(e.reward(),p,v,a,m)
                if e.isend():
                    break

    def isend(self,h=1):
        for i in self.creatures[0].phys:
            if i not in self.foot and i.p[1]<h+self.ground:
                return True
        return False

# ==================== 初始化 ====================

evnname="box2"
lastname="-deep-r11"
e=env()
statnum=len(e.getstat())
musclenum=sum([len(i.muscles) for i in e.creatures])
ground=e.ground
del e
savename=f"rlt-3-ppo-{evnname}{lastname}"
n=500   # 一轮训练的最大回合数
mode=1  # =1训练+显示，=0仅显示

# ==================== 训练子进程 ====================

def train_worker(queue):
    """训练子进程：持续训练并发送最佳模型到主进程"""
    m=model()
    mv=modelv()
    if os.path.exists(f"{savename}_policy.pth"):
        print("load",savename)
        m.load_state_dict(torch.load(f"{savename}_policy.pth",weights_only=True))
        mv.load_state_dict(torch.load(f"{savename}_value.pth",weights_only=True))

    opt_m=optim.Adam(m.parameters(),lr=0.001)
    opt_mv=optim.Adam(mv.parameters(),lr=0.001)
    memo=memory(2)
    ek=0.5
    ae=0.3
    best_r=-float('inf')
    save_count=0

    while True:
        try:
            ek=2 if ae<0.2 else 0.5
            r,al,av,ae=train(m,mv,opt_m,opt_mv,memo,discount=0.98,ek=ek,n=n)
        except (OverflowError,ZeroDivisionError,RuntimeError) as ex:
            r=0
            print(ex)
            memo.memo=[]
            continue

        torch.save(m.state_dict(),f"{savename}_policy.pth")
        torch.save(mv.state_dict(),f"{savename}_value.pth")
        save_count+=1
        if save_count%500==0:
            print(f"[git] {save_count} saves, pushing to main...")
            try:
                subprocess.run(["git","add",f"{savename}_policy.pth",f"{savename}_value.pth"],cwd=os.path.dirname(os.path.abspath(__file__))+ "/..",check=True)
                subprocess.run(["git","commit","-m",f"checkpoint {save_count}"],cwd=os.path.dirname(os.path.abspath(__file__))+ "/..",check=True)
                subprocess.run(["git","push","origin","main"],cwd=os.path.dirname(os.path.abspath(__file__))+ "/..",check=True)
                print("[git] push done")
            except subprocess.CalledProcessError as ex:
                print(f"[git] push failed: {ex}")

        if r>best_r:
            best_r=r
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
            queue.put(m.state_dict())

# ==================== 主进程 ====================

if __name__=='__main__':
    if mode:
        ctx=mp.get_context('spawn')
        queue=ctx.Queue(maxsize=2)
        p_train=ctx.Process(target=train_worker,args=(queue,))
        p_train.start()

        # 主进程：显示最佳模型动画
        m=model()
        while True:
            try:
                new_weights=queue.get_nowait()
                m.load_state_dict(new_weights)
                print("[display] updated best model")
            except:
                pass
            e=env()
            e.show(m)
    else:
        m=model()
        if os.path.exists(f"{savename}_policy.pth"):
            m.load_state_dict(torch.load(f"{savename}_policy.pth",weights_only=True))
        while True:
            e=env()
            e.show(m)
