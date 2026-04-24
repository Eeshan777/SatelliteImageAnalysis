import torch
import torch.nn as nn
import numpy as np
import os,cv2,uuid,json,datetime
from torchvision import transforms,models

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR=os.path.dirname(os.path.abspath(__file__))

class SatelliteAutoencoder(nn.Module):
	def __init__(self):
		super(SatelliteAutoencoder,self).__init__()
		self.encoder=nn.Sequential(
			nn.Conv2d(3,32,3,padding=1),nn.ReLU(),
			nn.MaxPool2d(2,2),
			nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
			nn.MaxPool2d(2,2)
		)
		self.decoder=nn.Sequential(
			nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
			nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),nn.ReLU(),
			nn.Conv2d(16,3,3,padding=1),nn.Sigmoid()
		)
	def forward(self,x):
		return self.decoder(self.encoder(x))
def get_vgg16_skeleton():
	model=models.vgg16(weights=None)
	model.avgpool=nn.AdaptiveAvgPool2d((1,1))
	model.classifier=nn.Sequential(
		nn.Flatten(),
		nn.Linear(512,256),
		nn.ReLU(True),
		nn.Dropout(0.5),
		nn.Linear(256,10)
	)
	return model
def load_pt_model(path,model_type="vgg"):
	if not os.path.exists(path):
		return None
	try:
		model=get_vgg16_skeleton() if model_type=="vgg" else SatelliteAutoencoder()
		state_dict=torch.load(path,map_location=device,weights_only=True)
		model.load_state_dict(state_dict)
		model.to(device).eval()
		return model
	except Exception as e:
		print(f"Error loading {model_type}: {e}")
		return None
FEATURE_MAP=[
	{"name":"Annual Crop","c":[255,255,0],"l":[20,40,40],"u":[35,255,255]},
	{"name":"Forest","c":[0,100,0],"l":[35,50,20],"u":[85,255,150]},
	{"name":"Herbaceous Veg","c":[173,255,47],"l":[25,30,50],"u":[45,150,255]},
	{"name":"Highway","c":[128,128,128],"l":[0,0,100],"u":[180,25,220]},
	{"name":"Industrial","c":[150,0,0],"l":[0,0,40],"u":[180,45,120]},
	{"name":"Pasture","c":[255,215,0],"l":[20,100,100],"u":[40,255,255]},
	{"name":"Permanent Crop","c":[34,139,34],"l":[40,40,20],"u":[70,255,100]},
	{"name":"Residential","c":[255,105,180],"l":[0,0,150],"u":[180,50,255]},
	{"name":"River","c":[0,191,255],"l":[90,50,50],"u":[110,255,200]},
	{"name":"Sea/Lake","c":[0,0,255],"l":[110,50,20],"u":[140,255,255]}
]
def log_to_json(full_path,label,confidence):
	output_folder=os.path.join(BASE_DIR,"outputs")
	os.makedirs(output_folder,exist_ok=True)
	report_path=os.path.join(output_folder,"report.json")
	entry={
		"file":full_path.replace("\\","/"),
		"label":label,
		"domain":"Satellite Image",
		"confidence":float(confidence),
		"timestamp":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
	}
	data=[]
	if os.path.exists(report_path):
		with open(report_path,"r") as f:
			try:
				data=json.load(f)
			except:
				data=[]
	data.append(entry)
	with open(report_path,"w") as f:
		json.dump(data,f,indent=4)
classifier=load_pt_model(os.path.join(BASE_DIR,"models","vgg16_model.pth"),"vgg")
autoencoder=load_pt_model(os.path.join(BASE_DIR,"models","autoencoder_model.pth"),"ae")
def analyze(path,grid_size=6):
	img_cv=cv2.imread(path)
	if img_cv is None:
		return None
	img_rgb=cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)
	to_tensor=transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
	])
	overlay=cv2.resize(img_rgb,(800,800))
	hsv_full=cv2.cvtColor(cv2.bilateralFilter(overlay,5,50,50),cv2.COLOR_RGB2HSV)
	for f in FEATURE_MAP:
		mask=cv2.inRange(hsv_full,np.array(f['l']),np.array(f['u']))
		overlay[mask>0]=cv2.addWeighted(overlay,0.7,np.full_like(overlay,f['c']),0.3,0)[mask>0]
	step=800//grid_size
	for i in range(1,grid_size):
		cv2.line(overlay,(i*step,0),(i*step,800),(0,0,0),2)
		cv2.line(overlay,(0,i*step),(800,i*step),(0,0,0),2)
	h,w,_=img_rgb.shape
	ph,pw=h//grid_size,w//grid_size
	patch_details=[]
	all_preds=[]
	classes=[f['name'] for f in FEATURE_MAP]
	with torch.no_grad():
		for r in range(grid_size):
			for c in range(grid_size):
				patch=img_rgb[r*ph:(r+1)*ph,c*pw:(c+1)*pw]
				if patch.size==0:
					continue
				tensor=to_tensor(patch).unsqueeze(0).to(device)
				if autoencoder:
					tensor=autoencoder(tensor)
					tensor=torch.nn.functional.interpolate(tensor,size=(224,224))
				output=classifier(tensor) if classifier else torch.zeros((1,10))
				p_preds=torch.softmax(output,dim=1).cpu().numpy()[0]
				all_preds.append(p_preds)
				hsv_p=cv2.cvtColor(patch,cv2.COLOR_RGB2HSV)
				feats=[]
				for f in FEATURE_MAP:
					m=cv2.inRange(hsv_p,np.array(f['l']),np.array(f['u']))
					cov=(np.sum(m>0)/(patch.shape[0]*patch.shape[1]))*100
					feats.append({
						"name":f['name'],
						"cov":f"{cov:.1f}%",
						"hex":'#%02x%02x%02x'%tuple(f['c'])
					})
				patch_details.append({
					"loc":f"[{r+1},{c+1}]",
					"vgg_label":classes[np.argmax(p_preds)],
					"vgg_conf":f"{np.max(p_preds)*100:.1f}%",
					"features":feats
				})
	mean_preds=np.mean(all_preds,axis=0)
	best_idx=np.argmax(mean_preds)
	log_to_json(path,classes[best_idx],mean_preds[best_idx])
	out_path=os.path.join(BASE_DIR,"outputs",f"{uuid.uuid4().hex}.png")
	cv2.imwrite(out_path,cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
	return {"global":classes[best_idx],"patches":patch_details,"seg_path":out_path}