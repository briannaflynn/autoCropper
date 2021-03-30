#!/usr/bin/python

def loadLearner(path_to_pkl):
	
	def acc(input, target):
		target = target.squeeze(1)
		return (input.argmax(dim=1)==target).float().mean()
		
	metrics = acc
	
	wd = 1e-2
	
	learn = load_learner(path_to_pkl)
	
	return learn
	
def pred2png(pred, in_file, out_dir = "./"):
    
    input = os.path.basename(in_file)
    out_file = input[:-4] + "_prediction.png"
    out_file = out_dir + out_file
    
    image = Image.open(in_file)
    shape = image.size
    
    x = pred[1]
    x = x.float()
    x_4 = torch.unsqueeze(x, 0)
    m = nn.Upsample(size = [shape[1], shape[0]], mode = 'nearest')
    j = m(x_4)
    
    j = j.numpy()
    j_int = j.astype(np.int)
    
    j = np.squeeze(j_int)
    
    
    cv.imwrite(out_file, j)