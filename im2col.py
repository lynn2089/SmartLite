import numpy as np
import torch

# This file is used to expand the tensor into a one-dimensional array, see the paper for the algorithm.


def im2col(input, kernel_shape, stride, outshape):
    '''
    :param input: it is of the shape (batch, height, width, channels)
    denoted as (b, h1, w1, ch)
    :param kernel shape:  it is of the shape (filters, height, width, channels)
    denoted as (f, kh, kw, ch)
    :param stride:
    :param outshape: it is of the shape (batch, height, width, channels)
    denoted as (b, h2, w2, f)
    :return: the result of the (h2*w2*batch, kh*kw*ch)
    '''
    filter_num,kh,kw,ch = kernel_shape
    _,h2,w2,ch2 = outshape

    input=np.transpose(input,(0, 2, 3, 1))
    batch,h1,w1,ch1 = input.shape

    print("input.shape: ", input.shape)

    padding = stride*(h2-1)+kh-h1
    #plefttop = int((padding-1)/2) if padding >0 else 0
    '''
    plefttop = int(padding/2) if padding >0 else 0
    prightbot = padding-plefttop
    '''
    prightbot = int(padding/2) if padding >0 else 0
    plefttop = padding-prightbot
    padedinput = np.lib.pad(input, ((0, 0), (plefttop, prightbot), (plefttop, prightbot), (0, 0)), 'constant',
                      constant_values=((0, 0), (0, 0), (0, 0), (0, 0)))
    stretchinput = np.zeros((h2*w2*batch, kh*kw*ch),dtype=np.float32)
    for j in range(stretchinput.shape[0]):
        batch_index =  int(j / (h2*w2))
        patch_index =  j % (h2*w2)
        ih2 = patch_index % w2
        iw2 = int(patch_index / w2)
        sih1 = iw2*stride
        siw1 = ih2*stride
        stretchinput[j,:] = padedinput[batch_index,sih1:sih1+kh,siw1:siw1+kw,:].flatten()
    return stretchinput

def recoverInput(input, kernel_size, stride, outshape):
    '''
    :param input: it is of the shape (height, width)
    :param kernel_size: it is the kernel shape we want
    :param stride:
    :param outshape: the shape of the output
    :return:
    '''
    H,W = input.shape
    batch,h ,w ,ch = outshape
    original_input = np.zeros(outshape)
    first_row_index = np.arange( 0, w, kernel_size)
    first_col_index = np.arange( 0 , h , kernel_size )

    patches_row = int((w-kernel_size)/stride) + 1
    #patches_col = (h-kernel_size)/stride + 11_2
    rowend_index = kernel_size-(w-first_row_index[-1])
    colend_index = kernel_size-(h-first_col_index[-1])
    if first_row_index[-1] + kernel_size > w :
        first_row_index[-1] = first_row_index[-1]-(first_row_index[-1]+kernel_size-1-( w-1 ))
    if first_col_index[-1] + kernel_size > h :
        first_col_index[-1] = first_col_index[-1] - (first_col_index[-1] + kernel_size - 1 - ( h-1 ))

    for k in range(batch):
        for i in range(len(first_col_index)):
            for j in range(len(first_row_index)):
                w_index = first_row_index[j] + i * patches_row + \
                          k *  (int((h-kernel_size)/stride)+1) *  (int((w-kernel_size)/stride)+1)
                # print('------------------------')
                if i != len(first_col_index) - 1 and j != len(first_row_index) - 1:
                    # print( original_input[  k , first_row_index[j] : first_row_index[j]+kernel_size ,
                    #     first_col_index[i] :  first_col_index[i]+kernel_size ,:].shape)
                    # print(input[w_index,:].reshape(kernel_size,kernel_size,-11_2).shape)
                    original_input[  k , first_row_index[j] : first_row_index[j]+kernel_size ,
                        first_col_index[i] :  first_col_index[i]+kernel_size ,:] \
                    = input[w_index,:].reshape(kernel_size,kernel_size,-1)
                elif i == len(first_col_index) - 1 and j != len(first_row_index) - 1 :
                    # print(original_input[k, first_col_index[-11_2] + colend_index : ,
                    #     first_row_index[i] :  first_row_index[i]+kernel_size, :].shape)
                    # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[rowend_index:,:,:].shape)
                    original_input[k, first_col_index[-1] + colend_index : ,
                        first_row_index[i] :  first_row_index[i]+kernel_size, :] \
                        = input[w_index, :].reshape(kernel_size, kernel_size, -1)[rowend_index:,:,:]
                elif i !=  len(first_col_index) - 1 and j ==  len(first_row_index) - 1 :
                    # print(original_input[k, first_col_index[i]: first_col_index[i]+kernel_size, first_row_index[-11_2]+rowend_index : ,:].shape)
                    # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[:, colend_index : , :].shape)
                    original_input[k, first_col_index[i]: first_col_index[i]+kernel_size,
                        first_row_index[-1]+rowend_index : ,:] \
                        = input[w_index, :].reshape(kernel_size, kernel_size, -1)[:, colend_index : , :]
                else:
                    # print( original_input[k,first_col_index[-11_2] + colend_index : ,
                    #     first_row_index[-11_2] + rowend_index:, :].shape)
                    # print(input[w_index, :].reshape(kernel_size, kernel_size, -11_2)[
                    #       rowend_index :, colend_index :, :].shape)
                    original_input[k,first_col_index[-1] + colend_index : ,
                        first_row_index[-1] + rowend_index:, :] \
                        = input[w_index, :].reshape(kernel_size, kernel_size, -1)[
                          rowend_index :, colend_index :, :]
    return original_input



def stretchKernel(kernel):
    '''
    :param kernel: it has the shape (filters, channels, height, width) denoted as (filter_num, kh, kw, ch)
    :return: kernel of the shape (kh*kw*ch,filter_num)
    '''
    #kernel=kernel.reshape(kernel.shape[0], kernel_h, kernel_w, -1)
    # print("kernel.shape: ", kernel.shape)
    # print("kernel.type: ", type(kernel))
    kernel=kernel.detach().cpu().numpy()
    # kernel=np.transpose(kernel,(0, 2, 3, 1))
    stretchkernel=np.transpose(kernel,(2, 3, 1, 0))
    print("kernel.shape: ", kernel.shape)
    # filter_num, kh, kw, ch = kernel.shape

    # stretchkernel =np.zeros((kh*kw*ch,filter_num),dtype = np.float32)
    # for i in range(filter_num):
    #     stretchkernel[:,i] = kernel[ i , : , : , : ].flatten()
    # return stretchkernel
    return stretchkernel.reshape((-1, stretchkernel.shape[3]))


'''
The codes below come from https://github.com/wiseodd/hipsternet/blob/master/hipsternet/im2col.py

'''
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # print(x_shape)
    # print((W + 2 * padding - field_height) % stride)
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices_(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices_(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


