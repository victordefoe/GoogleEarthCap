# encoding: utf-8
'''
@Author: 刘琛
@Time: 2019/11/13 1:23
@Contact: victordefoe88@gmail.com

@File: backup.py
@Statement:
This is for the useless backup codes, just for convenient further reading
'''

# ----  tring thirdparty chinese_ocr app , but it fails ----
# from thirdparty.chineseocr_app.crnn.network_torch import CRNN
# from thirdparty.chineseocr_app.crnn.keys import alphabetChinese, alphabetEnglish
# ocrModelWeight = os.path.join(pwd, "thirdparty","chineseocr_app", "models", "ocr-lstm.pth")
# alphabet = alphabetChinese
# nclass = len(alphabet)+1
# LSTMFLAG = True
# GPU = False
# OCRMODEL = CRNN( 32, 1, nclass, 256, leakyRelu=False,lstmFlag=LSTMFLAG,GPU=GPU,alphabet=alphabet)
# OCRMODEL.load_weights(ocrModelWeight)


##

#
# class parmeters():
#     def __init__(self):
#         self.image_folder = 1
#         self.workers = 2
#         self.batch_size = 1
#         self.saved_model = 'D://UAV_location//google_earth//GooleEarth//thirdparty' \
#                            '//deep_tr//pretrained_models//TPS-ResNet-BiLSTM-Attn.pth'
#         self.batch_max_length = 25
#         self.imgH = 32
#         self.imgW = 100
#         self.rgb = False
#         self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
#         self.sensitive = True
#         self.PAD = True
#         self.Transformation = 'TPS'
#         self.FeatureExtraction = 'ResNet'
#         self.SequenceModeling = 'BiLSTM'
#         self.Prediction = 'Attn'
#         self.num_fiducial = 20
#         self.input_channel = 1
#         self.output_channel = 512
#         self.hidden_size = 256
#
#
# opt = parmeters()
# print(opt.FeatureExtraction)
#
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# converter = AttnLabelConverter(opt.character)
# opt.num_class = len(converter.character)
# if opt.rgb:
#     opt.input_channel = 3
# model = Model(opt).to(device)
# print(
#     'model input parameters',
#     opt.imgH,
#     opt.imgW,
#     opt.num_fiducial,
#     opt.input_channel,
#     opt.output_channel,
#     opt.hidden_size,
#     opt.num_class,
#     opt.batch_max_length,
#     opt.Transformation,
#     opt.FeatureExtraction,
#     opt.SequenceModeling,
#     opt.Prediction)
#
# # model = torch.nn.DataParallel(model).to(device)
# # load model
#
#
# print('loading pretrained model from %s' % opt.saved_model)
# state_dict = torch.load(opt.saved_model)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)
#
# ###
#
# from thirdparty.deep_tr.dataset import ResizeNormalize
# transform = ResizeNormalize((opt.imgW, opt.imgH))
# image_tensor = transform(img).unsqueeze(0)
# ###
#
# # with torchsnooper.snoop():
# # image_tensor = img
# print('..........',image_tensor.size())
# image = image_tensor.to(device)
# text_for_pred = torch.LongTensor(
#     opt.batch_size,
#     opt.batch_max_length +
#     1).fill_(0).to(device)
#
#
# preds = model(image, text_for_pred, is_train=False)
# _, preds_index = preds.max(2)
# length_for_pred = torch.IntTensor([opt.batch_max_length] * opt.batch_size).to(device)
# preds_str = converter.decode(preds_index, length_for_pred)
#
# print(preds_str)


# print(tesserocr.tesseract_version())  # print tesseract-ocr version
# prints tessdata path and list of available languages
# print(tesserocr.get_languages())

# with PyTessBaseAPI() as api:
#     api.SetImageFile('loc.jpg')
#     print(api.GetUTF8Text())
#     print(api.AllWordConfidences())

# print(tesserocr.file_to_text('loc.jpg', lang='Armenian', psm=7 ))
