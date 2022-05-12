import torch
import math
import torch.nn as nn

class Decoder():
    def __init__(self, tokenizer, word2ix, model, device="cuda"):
        super().__init__()
        self.tokenizer = tokenizer
        self.word2ix = word2ix
        self.model = model
        self.device = device
        self.out_max_length = 30
        # output_model_file = os.path.join(args.save_path, "model.bin")
        # self.model.load_state_dict(torch.load(output_model_file, map_location=self.device)['model_state_dict'])

    def generate(self, input_ids, token_type_ids, beam_size=1):
        raise NotImplementedError()


    def beam_search(self, token_ids, token_type_ids, word2ix, beam_size=1):
        sep_id = word2ix["[SEP]"]
        # 用来保存输出序列
        output_ids = [[]]
        # 用来保存累计得分
        output_scores = torch.zeros(token_ids.shape[0], device=self.device)
        for step in range(self.out_max_length):
            scores = self.model.forward(token_ids, token_type_ids)   #  [bs, len, vocab_size]
            if step == 0:
                # 重复beam-size次 输入ids
                token_ids = token_ids.view(1, -1).repeat(beam_size, 1)
                token_type_ids = token_type_ids.view(1, -1).repeat(beam_size, 1)
            ## 计算log 分值 (beam_size, vocab_size)
            logit_score = torch.log_softmax(scores, dim=-1)[:, -1]
            logit_score = output_scores.view(-1, 1) + logit_score  # 累计得分
            ## 取topk的时候我们是展平了然后再去调用topk函数
            # 展平
            logit_score = logit_score.view(-1)
            hype_score, hype_pos = torch.topk(logit_score, beam_size)
            indice1 = hype_pos / scores.shape[-1]  # 行索引
            indice2 = hype_pos % scores.shape[-1]  # 列索引

            # 下面需要更新一下输出了
            new_hype_scores = []
            new_hype_ids = []
            # 为啥有这个[],就是因为要过滤掉结束的序列。
            next_chars = []  # 用来保存新预测出来的一个字符，继续接到输入序列后面，再去预测新字符
            for i_1, i_2, score in zip(indice1, indice2, hype_score):
                i_1 = int(i_1.item())
                i_2 = int(i_2.item())

                hype_id = output_ids[i_1] + [i_2]  # 保存所有输出的序列，而不仅仅是新预测的单个字符

                if i_2 == sep_id:
                    # 说明解码到最后了
                    if score == torch.max(hype_score).item():
                        # 说明找到得分最大的那个序列了 直接返回即可
                        return hype_id[: -1]
                    else:
                        # 完成一个解码了，但这个解码得分并不是最高，因此的话需要跳过这个序列
                        beam_size -= 1
                else:
                    new_hype_ids.append(hype_id)
                    new_hype_scores.append(score)
                    next_chars.append(i_2)  # 收集一下，需要连接到当前的输入序列之后

            output_ids = new_hype_ids

            output_scores = torch.tensor(new_hype_scores, dtype=torch.float32, device=self.device)
            # 现在需要重新构造输入数据了，用上一次输入连接上这次新输出的字符，再输入bert中预测新字符
            token_ids = token_ids[:len(output_ids)].contiguous()  # 截取，因为要过滤掉已经完成预测的序列
            token_type_ids = token_type_ids[: len(output_ids)].contiguous()

            next_chars = torch.tensor(next_chars, dtype=torch.long, device=self.device).view(-1, 1)
            next_token_type_ids = torch.ones_like(next_chars, device=self.device)
            # 连接
            token_ids = torch.cat((token_ids, next_chars), dim=1)
            token_type_ids = torch.cat((token_type_ids, next_token_type_ids), dim=1)
            if beam_size < 1:
                break

        # 如果达到最大长度的话 直接把得分最高的输出序列返回把
        return output_ids[output_scores.argmax().item()]


class AutoRegressiveDecoder(object):
    """通用自回归生成模型解码基类
    包含beam search和random sample两种策略
    """
    def __init__(self, start_id, end_id, maxlen, minlen=1, device='cpu'):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.models = {}
        self.device = device
        if start_id is None:
            self.first_output_ids = torch.empty((1, 0), dtype=int, device=device)
        else:
            self.first_output_ids = torch.tensor([[self.start_id]], device=device)

    @staticmethod
    def wraps(default_rtype='probas', use_states=False):
        """用来进一步完善predict函数
        目前包含: 1. 设置rtype参数，并做相应处理；
                  2. 确定states的使用，并做相应处理；
                  3. 设置温度参数，并做相应处理。
        """
        def actual_decorator(predict):
            def new_predict(
                self,
                inputs,
                output_ids,
                states,
                temperature=1,
                rtype=default_rtype
            ):
                assert rtype in ['probas', 'logits']
                prediction = predict(self, inputs, output_ids, states)

                if not use_states:
                    prediction = (prediction, None)

                if default_rtype == 'logits':
                    prediction = (nn.Softmax(dim=-1)(prediction[0] / temperature), prediction[1])
                elif temperature != 1:
                    probas = torch.power(prediction[0], 1.0 / temperature)
                    probas = probas / probas.sum(axis=-1, keepdims=True)
                    prediction = (probas, prediction[1])

                if rtype == 'probas':
                    return prediction
                else:
                    return torch.log(prediction[0] + 1e-12), prediction[1]

            return new_predict

        return actual_decorator

    # def last_token(self, model):
    #     """创建一个只返回最后一个token输出的新Model
    #     """
    #     if model not in self.models:
    #         outputs = [
    #             keras.layers.Lambda(lambda x: x[:, -1])(output)
    #             for output in model.outputs
    #         ]
    #         self.models[model] = keras.models.Model(model.inputs, outputs)

    #     return self.models[model]

    def predict(self, inputs, output_ids, states=None):
        """用户需自定义递归预测函数
        说明: 定义的时候，需要用wraps方法进行装饰，传入default_rtype和use_states，
             其中default_rtype为字符串logits或probas，probas时返回归一化的概率，
             rtype=logits时则返回softmax前的结果或者概率对数。
        返回: 二元组 (得分或概率, states)
        """
        raise NotImplementedError

    def beam_search(self, inputs_raw, topk, states=None, temperature=1, min_ends=1, add_btz_dim=True):
        """beam search解码
        说明: 这里的topk即beam size；
        返回: 最优解码序列。
        """
        inputs = []
        for i in inputs_raw:
            if isinstance(i, torch.torch.Tensor):
                pass
            elif isinstance(i, (list, tuple, np.ndarray)) and add_btz_dim:
                i = torch.tensor([i], device=self.device)
            elif isinstance(i, (list, tuple, np.ndarray)) and not add_btz_dim:
                i = torch.tensor(i, device=self.device)
            else:
                raise ValueError('Beam search inputs ele only support tensor、array、list、tuple')
            inputs.append(i)

        output_ids, output_scores = self.first_output_ids, torch.zeros(1, device=self.device)
        for step in range(self.maxlen):
            scores, states = self.predict(inputs, output_ids, states, temperature, 'logits')  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [i.repeat([topk]+[1]*(len(i.shape)-1)) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.flatten().argsort(dim=-1, descending=True)[:topk]  # 仅保留topk
            indices_1 = torch.div(indices, scores.shape[1], rounding_mode='trunc')  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = torch.cat([output_ids[indices_1], indices_2], 1)  # 更新输出
            output_scores = torch.take_along_dim(scores, indices, dim=None)  # 更新得分
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best = output_scores.argmax()  # 得分最大的那个
                if is_end[best] and end_counts[best] >= min_ends:  # 如果已经终止
                    return output_ids[best]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = ~is_end | (end_counts < min_ends)  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
        # 达到长度直接输出
        return output_ids[output_scores.argmax()]

    def random_sample(
        self,
        inputs,
        n,
        topk=None,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1
    ):
        """随机采样n个结果
        说明: 非None的topk表示每一步只从概率最高的topk个中采样；而非None的topp
             表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样。
        返回: n个解码序列组成的list。
        """
        inputs = [torch.tensor([i], device=self.device) for i in inputs]
        output_ids = self.first_output_ids
        results = []
        for step in range(self.maxlen):
            probas, states = self.predict(inputs, output_ids, states, temperature, 'probas')  # 计算当前概率
            probas /= probas.sum(dim=-1, keepdims=True)  # 确保归一化
            if step == 0:  # 第1步预测后将结果重复n次
                probas = probas.repeat([n]+[1]*(len(probas.shape)-1))
                inputs = [i.repeat([n]+[1]*(len(i.shape)-1)) for i in inputs]
                output_ids = output_ids.repeat([n]+[1]*(len(output_ids.shape)-1))
            if topk is not None:
                k_indices = probas.argsort(dim=-1, descending=True)[:, :topk]  # 仅保留topk
                probas = torch.take_along_dim(probas, k_indices, dim=1)  # topk概率
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化
            if topp is not None:
                p_indices = probas.argsort(dim=-1, descending=True)  # 从高到低排序
                probas = torch.take_along_dim(probas, p_indices, dim=-1)  # 排序概率
                cumsum_probas = torch.cumsum(probas, dim=-1)  # 累积概率
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)  # 标记超过topp的部分
                flag[:, 0] = False  # 结合上面的torch.roll，实现平移一位的效果
                probas[flag] = 0  # 后面的全部置零
                probas /= probas.sum(dim=1, keepdims=True)  # 重新归一化

            sample_func = lambda p: torch.multinomial(p, 1)  # 按概率采样函数
            sample_ids = torch.stack([sample_func(p) for p in probas])
            sample_ids = sample_ids.reshape((-1, 1))  # 对齐形状
            if topp is not None:
                sample_ids = torch.take_along_dim(p_indices, sample_ids, dim=1)  # 对齐原id
            if topk is not None:
                sample_ids = torch.take_along_dim(k_indices, sample_ids, dim=1)  # 对齐原id
            output_ids = torch.cat([output_ids, sample_ids], 1)  # 更新输出
            is_end = output_ids[:, -1] == self.end_id  # 标记是否以end标记结束
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                flag = is_end & (end_counts >= min_ends)  # 标记已完成序列
                if flag.any():  # 如果有已完成的
                    for ids in output_ids[flag]:  # 存好已完成序列
                        results.append(ids)
                    flag = (flag == False)  # 标记未完成序列
                    inputs = [i[flag] for i in inputs]  # 只保留未完成部分输入
                    output_ids = output_ids[flag]  # 只保留未完成部分候选集
                    end_counts = end_counts[flag]  # 只保留未完成部分end计数
                    if len(output_ids) == 0:
                        break
        # 如果还有未完成序列，直接放入结果
        for ids in output_ids:
            results.append(ids)
        # 返回结果
        return results


def Bertpooling(bert_outputs, pool_type='pooler'):
    encoded_layers = bert_outputs.hidden_states
    sequence_output = bert_outputs.last_hidden_state

    if pool_type == 'pooler':
        pooled_output = bert_outputs.pooler_output

    elif pool_type == 'first-last-avg':
        first = encoded_layers[1]
        last = encoded_layers[-1]
        seq_length = first.size(1)

        first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
        last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  # batch, hid_size
        pooled_output = torch.avg_pool1d(
            torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
            kernel_size=2).squeeze(-1)

    elif pool_type == 'last-avg':
        pooled_output = torch.mean(sequence_output, 1)

    elif pool_type == 'cls':
        pooled_output = sequence_output[:, 0]

    else:
        raise TypeError('Please the right pool_type from cls, pooler, first-last-avg and last-avg !!!')

    return pooled_output


def unilm_mask(token_type_ids):
    seq_len = token_type_ids.size(1)
    bs = token_type_ids.size(0)
    idxs = torch.cumsum(token_type_ids, dim=1)
    mask = (idxs.unsqueeze(1) <= idxs.unsqueeze(2)).long()

    last_one_idx = 0
    for bsz in range(bs):
        for i, v in enumerate(token_type_ids[bsz]):
            if v == 1:
                last_one_idx = i
        pad_mask = torch.tensor([1]*(last_one_idx+1) + [0]* (seq_len-last_one_idx-1))
        pad_mask = (pad_mask.unsqueeze(0) * pad_mask.unsqueeze(1)).to(mask.device)
        mask[bsz] = mask[bsz]*pad_mask
        last_one_idx = 0

    return mask


def gelu(x):
    """
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


def build_bert_optim_parameters(model, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    return optimizer_grouped_parameters
