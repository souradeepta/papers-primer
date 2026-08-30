"""Toy PagedAttention-style logical KV block table with shared-prefix refs."""
class Manager:
 def __init__(self,size=4): self.size=size;self.refs={};self.next=0
 def alloc(self): b=self.next;self.next+=1;self.refs[b]=1;return b
 def share(self,b): self.refs[b]+=1;return b
 def free(self,b):
  self.refs[b]-=1
  if not self.refs[b]: del self.refs[b]
 def locate(self,table,index): return table[index//self.size],index%self.size
def main():
 m=Manager();prefix=m.alloc();a=[prefix,m.alloc()];b=[m.share(prefix),m.alloc()]
 assert m.locate(a,5)==(a[1],1) and m.refs[prefix]==2
 m.free(a[0]);m.free(a[1]);assert prefix in m.refs
 m.free(b[0]);m.free(b[1]);assert not m.refs
 print('ok: logical blocks translate correctly and a shared prefix survives until its final release')
if __name__=='__main__':main()
