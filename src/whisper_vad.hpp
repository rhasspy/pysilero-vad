#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#define _USE_MATH_DEFINES
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define WHISPER_MAX_NODES 4096

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX //  prevent min/max macros
#include <windows.h>

static std::wstring utf8_to_wstring(const char *utf8) {
    if (!utf8) {
        return std::wstring();
    }

    int len = MultiByteToWideChar(CP_UTF8, 0, utf8,
                                  -1, // input is NUL-terminated
                                  nullptr, 0);
    if (len <= 0) {
        return std::wstring();
    }

    std::wstring w(len - 1, L'\0'); // exclude terminating NUL
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, w.data(), len - 1);
    return w;
}
#endif

enum vad_tensor {
    VAD_TENSOR_STFT_BASIS,
    VAD_TENSOR_ENC_0_WEIGHT,
    VAD_TENSOR_ENC_0_BIAS,
    VAD_TENSOR_ENC_1_WEIGHT,
    VAD_TENSOR_ENC_1_BIAS,
    VAD_TENSOR_ENC_2_WEIGHT,
    VAD_TENSOR_ENC_2_BIAS,
    VAD_TENSOR_ENC_3_WEIGHT,
    VAD_TENSOR_ENC_3_BIAS,
    VAD_TENSOR_LSTM_WEIGHT_IH,
    VAD_TENSOR_LSTM_WEIGHT_HH,
    VAD_TENSOR_LSTM_BIAS_IH,
    VAD_TENSOR_LSTM_BIAS_HH,
    VAD_TENSOR_FINAL_CONV_WEIGHT,
    VAD_TENSOR_FINAL_CONV_BIAS,
};

static const std::map<vad_tensor, ggml_op> VAD_TENSOR_OPS = {
    {VAD_TENSOR_STFT_BASIS, GGML_OP_IM2COL},
    {VAD_TENSOR_ENC_0_WEIGHT, GGML_OP_IM2COL},
    {VAD_TENSOR_ENC_0_BIAS, GGML_OP_ADD},
    {VAD_TENSOR_ENC_1_WEIGHT, GGML_OP_IM2COL},
    {VAD_TENSOR_ENC_1_BIAS, GGML_OP_ADD},
    {VAD_TENSOR_ENC_2_WEIGHT, GGML_OP_IM2COL},
    {VAD_TENSOR_ENC_2_BIAS, GGML_OP_ADD},
    {VAD_TENSOR_ENC_3_WEIGHT, GGML_OP_IM2COL},
    {VAD_TENSOR_ENC_3_BIAS, GGML_OP_ADD},

    {VAD_TENSOR_LSTM_WEIGHT_IH, GGML_OP_MUL_MAT},
    {VAD_TENSOR_LSTM_WEIGHT_HH, GGML_OP_MUL_MAT},
    {VAD_TENSOR_LSTM_BIAS_IH, GGML_OP_ADD},
    {VAD_TENSOR_LSTM_BIAS_HH, GGML_OP_ADD},

    {VAD_TENSOR_FINAL_CONV_WEIGHT, GGML_OP_IM2COL},
    {VAD_TENSOR_FINAL_CONV_BIAS, GGML_OP_ADD}};

static const std::map<vad_tensor, const char *> VAD_TENSOR_NAMES = {
    {VAD_TENSOR_STFT_BASIS, "_model.stft.forward_basis_buffer"},
    {VAD_TENSOR_ENC_0_WEIGHT, "_model.encoder.0.reparam_conv.weight"},
    {VAD_TENSOR_ENC_0_BIAS, "_model.encoder.0.reparam_conv.bias"},
    {VAD_TENSOR_ENC_1_WEIGHT, "_model.encoder.1.reparam_conv.weight"},
    {VAD_TENSOR_ENC_1_BIAS, "_model.encoder.1.reparam_conv.bias"},
    {VAD_TENSOR_ENC_2_WEIGHT, "_model.encoder.2.reparam_conv.weight"},
    {VAD_TENSOR_ENC_2_BIAS, "_model.encoder.2.reparam_conv.bias"},
    {VAD_TENSOR_ENC_3_WEIGHT, "_model.encoder.3.reparam_conv.weight"},
    {VAD_TENSOR_ENC_3_BIAS, "_model.encoder.3.reparam_conv.bias"},
    {VAD_TENSOR_LSTM_WEIGHT_IH, "_model.decoder.rnn.weight_ih"},
    {VAD_TENSOR_LSTM_WEIGHT_HH, "_model.decoder.rnn.weight_hh"},
    {VAD_TENSOR_LSTM_BIAS_IH, "_model.decoder.rnn.bias_ih"},
    {VAD_TENSOR_LSTM_BIAS_HH, "_model.decoder.rnn.bias_hh"},
    {VAD_TENSOR_FINAL_CONV_WEIGHT, "_model.decoder.decoder.2.weight"},
    {VAD_TENSOR_FINAL_CONV_BIAS, "_model.decoder.decoder.2.bias"}};

typedef struct whisper_model_loader {
    void *context;

    size_t (*read)(void *ctx, void *output, size_t read_size);
    bool (*eof)(void *ctx);
    void (*close)(void *ctx);
} whisper_model_loader;

template <typename T>
static void read_safe(whisper_model_loader *loader, T &dest) {
    loader->read(loader->context, &dest, sizeof(T));
}

struct whisper_vad_hparams {
    int32_t n_encoder_layers;
    int32_t *encoder_in_channels;
    int32_t *encoder_out_channels;
    int32_t *kernel_sizes;
    int32_t lstm_input_size;
    int32_t lstm_hidden_size;
    int32_t final_conv_in;
    int32_t final_conv_out;
};

struct whisper_vad_context_params {
    int n_threads; // The number of threads to use for processing.
};

struct whisper_vad_model {
    std::string type;
    std::string version;
    whisper_vad_hparams hparams;

    struct ggml_tensor *stft_forward_basis; // [256, 1, 258]

    // Encoder tensors - 4 convolutional layers
    struct ggml_tensor *encoder_0_weight; // [3, 129, 128]
    struct ggml_tensor *encoder_0_bias;   // [128]

    // Second encoder layer
    struct ggml_tensor *encoder_1_weight; // [3, 128, 64]
    struct ggml_tensor *encoder_1_bias;   // [64]

    // Third encoder layer
    struct ggml_tensor *encoder_2_weight; // [3, 64, 64]
    struct ggml_tensor *encoder_2_bias;   // [64]

    // Fourth encoder layer
    struct ggml_tensor *encoder_3_weight; // [3, 64, 128]
    struct ggml_tensor *encoder_3_bias;   // [128]

    // LSTM decoder tensors
    struct ggml_tensor *lstm_ih_weight; // [128, 512] input-to-hidden
    struct ggml_tensor *lstm_ih_bias;   // [512]
    struct ggml_tensor *lstm_hh_weight; // [128, 512] hidden-to-hidden
    struct ggml_tensor *lstm_hh_bias;   // [512]

    // Final conv layer
    struct ggml_tensor *final_conv_weight; // [128]
    struct ggml_tensor *final_conv_bias;   // [1]

    // ggml contexts
    std::vector<ggml_context *> ctxs;

    // buffer for the model tensors
    std::vector<ggml_backend_buffer_t> buffers;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct whisper_sched {
    ggml_backend_sched_t sched = nullptr;

    std::vector<uint8_t> meta;
};

struct whisper_vad_context {
    int64_t t_vad_us = 0;

    int n_window;
    int n_context;
    int n_threads;

    std::vector<ggml_backend_t> backends;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<uint8_t> ctx_buf;
    whisper_sched sched;

    whisper_vad_model model;
    std::string path_model;
    struct ggml_tensor *h_state;
    struct ggml_tensor *c_state;

    ggml_cgraph *gf = nullptr;
};

void whisper_vad_free(whisper_vad_context *ctx) {
    if (ctx) {
        if (ctx->buffer) {
            ggml_backend_buffer_free(ctx->buffer);
        }
        for (ggml_context *context : ctx->model.ctxs) {
            ggml_free(context);
        }

        for (ggml_backend_buffer_t buf : ctx->model.buffers) {
            ggml_backend_buffer_free(buf);
        }

        ggml_backend_sched_free(ctx->sched.sched);

        for (auto &backend : ctx->backends) {
            ggml_backend_free(backend);
        }

        delete[] ctx->model.hparams.encoder_in_channels;
        delete[] ctx->model.hparams.encoder_out_channels;
        delete[] ctx->model.hparams.kernel_sizes;

        delete ctx;
    }
}

struct whisper_vad_context_params whisper_vad_default_context_params(void) {
    whisper_vad_context_params result = {
        /*.n_thread                = */ 4,
    };
    return result;
}

// measure the memory usage of a graph and prepare the allocr's internal data
// buffer
static bool
whisper_sched_graph_init(struct whisper_sched &allocr,
                         std::vector<ggml_backend_t> backends,
                         std::function<struct ggml_cgraph *()> &&get_graph) {
    auto &sched = allocr.sched;
    auto &meta = allocr.meta;

    sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(),
                                   WHISPER_MAX_NODES, false, true);

    meta.resize(ggml_tensor_overhead() * WHISPER_MAX_NODES +
                ggml_graph_overhead());

    // since there are dependencies between the different graphs,
    // we need to allocate them instead of only reserving to get the correct
    // compute buffer size
    if (!ggml_backend_sched_alloc_graph(sched, get_graph())) {
        // failed to allocate the compute buffer
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

static ggml_tensor *whisper_vad_build_stft_layer(ggml_context *ctx0,
                                                 const whisper_vad_model &model,
                                                 ggml_tensor *cur) {
    // Apply reflective padding to the input tensor
    ggml_tensor *padded = ggml_pad_reflect_1d(ctx0, cur, 64, 64);

    struct ggml_tensor *stft =
        ggml_conv_1d(ctx0, model.stft_forward_basis, padded,
                     model.hparams.lstm_input_size, 0, 1);

    // Calculate cutoff for real/imaginary parts
    int cutoff = model.stft_forward_basis->ne[2] / 2;

    // Extract real part (first half of the STFT output).
    struct ggml_tensor *real_part =
        ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1], 0);
    // Extract imaginary part (second half of the STFT output).
    struct ggml_tensor *img_part =
        ggml_view_2d(ctx0, stft, 4, cutoff, stft->nb[1], cutoff * stft->nb[1]);

    // Calculate magnitude: sqrt(real^2 + imag^2)
    struct ggml_tensor *real_squared = ggml_mul(ctx0, real_part, real_part);
    struct ggml_tensor *img_squared = ggml_mul(ctx0, img_part, img_part);
    struct ggml_tensor *sum_squares = ggml_add(ctx0, real_squared, img_squared);
    struct ggml_tensor *magnitude = ggml_sqrt(ctx0, sum_squares);
    return magnitude;
}

static ggml_tensor *whisper_vad_build_encoder_layer(
    ggml_context *ctx0, const whisper_vad_model &model, ggml_tensor *cur) {
    // First Conv1D: expands to 128 channels.
    cur = ggml_conv_1d(ctx0, model.encoder_0_weight, cur, 1, 1, 1);
    cur = ggml_add(ctx0, cur,
                   ggml_reshape_3d(ctx0, model.encoder_0_bias, 1, 128, 1));
    cur = ggml_relu(ctx0, cur);

    // Second Conv1D: reduces to 64 channels.
    cur = ggml_conv_1d(ctx0, model.encoder_1_weight, cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur,
                   ggml_reshape_3d(ctx0, model.encoder_1_bias, 1, 64, 1));
    cur = ggml_relu(ctx0, cur);

    // Third Conv1D: maintains 64 channels
    cur = ggml_conv_1d(ctx0, model.encoder_2_weight, cur, 2, 1, 1);
    cur = ggml_add(ctx0, cur,
                   ggml_reshape_3d(ctx0, model.encoder_2_bias, 1, 64, 1));
    cur = ggml_relu(ctx0, cur);

    // Fourth Conv1D: expands to 128 channels
    cur = ggml_conv_1d(ctx0, model.encoder_3_weight, cur, 1, 1, 1);
    cur = ggml_add(ctx0, cur,
                   ggml_reshape_3d(ctx0, model.encoder_3_bias, 1, 128, 1));
    cur = ggml_relu(ctx0, cur);

    return cur;
}

static ggml_tensor *
whisper_vad_build_lstm_layer(ggml_context *ctx0,
                             const whisper_vad_context &vctx, ggml_tensor *cur,
                             ggml_cgraph *gf) {
    const whisper_vad_model &model = vctx.model;
    const int hdim = model.hparams.lstm_hidden_size;

    struct ggml_tensor *x_t = ggml_transpose(ctx0, cur);

    // Create operations using the input-to-hidden weights.
    struct ggml_tensor *inp_gate =
        ggml_mul_mat(ctx0, model.lstm_ih_weight, x_t);
    inp_gate = ggml_add(ctx0, inp_gate, model.lstm_ih_bias);

    // Create operations using the hidden-to-hidden weights.
    struct ggml_tensor *hid_gate =
        ggml_mul_mat(ctx0, model.lstm_hh_weight, vctx.h_state);
    hid_gate = ggml_add(ctx0, hid_gate, model.lstm_hh_bias);

    // Create add operation to get preactivations for all gates.
    struct ggml_tensor *out_gate = ggml_add(ctx0, inp_gate, hid_gate);

    const size_t hdim_size = ggml_row_size(out_gate->type, hdim);

    // Create sigmoid for input gate (using the first 128 bytes from the
    // preactivations).
    struct ggml_tensor *i_t =
        ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 0 * hdim_size));

    // Create sigmoid for the forget gate (using the second 128 bytes from the
    // preactivations).
    struct ggml_tensor *f_t =
        ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 1 * hdim_size));

    // Create sigmoid for the cell gate (using the third 128 bytes from the
    // preactivations).
    struct ggml_tensor *g_t =
        ggml_tanh(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 2 * hdim_size));

    // Create sigmoid for the output gate (using the fourth 128 bytes from the
    // preactivations).
    struct ggml_tensor *o_t =
        ggml_sigmoid(ctx0, ggml_view_1d(ctx0, out_gate, hdim, 3 * hdim_size));

    // Update cell state
    struct ggml_tensor *c_out = ggml_add(
        ctx0, ggml_mul(ctx0, f_t, vctx.c_state), ggml_mul(ctx0, i_t, g_t));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, c_out, vctx.c_state));

    // Update hidden state
    struct ggml_tensor *out = ggml_mul(ctx0, o_t, ggml_tanh(ctx0, c_out));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, out, vctx.h_state));

    return out;
}

static struct ggml_cgraph *whisper_vad_build_graph(whisper_vad_context &vctx) {
    const auto &model = vctx.model;

    struct ggml_init_params params = {
        /*.mem_size   =*/vctx.sched.meta.size(),
        /*.mem_buffer =*/vctx.sched.meta.data(),
        /*.no_alloc   =*/true,
    };

    struct ggml_context *ctx0 = ggml_init(params);

    ggml_cgraph *gf = ggml_new_graph(ctx0);

    struct ggml_tensor *frame =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, vctx.n_window, 1);
    ggml_set_name(frame, "frame");
    ggml_set_input(frame);

    struct ggml_tensor *cur = nullptr;
    {
        cur = whisper_vad_build_stft_layer(ctx0, model, frame);

        cur = whisper_vad_build_encoder_layer(ctx0, model, cur);

        // Extract the first element of the first dimension
        // (equivalent to pytorch's [:, :, 0])
        cur = ggml_view_2d(ctx0, cur, 1, 128, cur->nb[1], 0);

        cur = whisper_vad_build_lstm_layer(ctx0, vctx, cur, gf);
        cur = ggml_relu(ctx0, cur);
        cur = ggml_conv_1d(ctx0, model.final_conv_weight, cur, 1, 0, 1);
        cur = ggml_add(ctx0, cur, model.final_conv_bias);
        cur = ggml_sigmoid(ctx0, cur);
        ggml_set_name(cur, "prob");
        ggml_set_output(cur);
    }

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);

    return gf;
}

static bool whisper_vad_init_context(whisper_vad_context *vctx) {
    vctx->backends.push_back(
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL));

    const int32_t lstm_hidden_size = vctx->model.hparams.lstm_hidden_size;

    vctx->ctx_buf.resize(2u * ggml_tensor_overhead());

    struct ggml_init_params params = {
        /*.mem_size   =*/vctx->ctx_buf.size(),
        /*.mem_buffer =*/vctx->ctx_buf.data(),
        /*.no_alloc   =*/true,
    };

    ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    // LSTM Hidden state
    vctx->h_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, lstm_hidden_size);
    ggml_set_name(vctx->h_state, "h_state");

    // LSTM Cell state
    vctx->c_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, lstm_hidden_size);
    ggml_set_name(vctx->c_state, "c_state");

    vctx->buffer = ggml_backend_alloc_ctx_tensors(ctx, vctx->backends[0]);
    ggml_free(ctx);
    if (!vctx->buffer) {
        return false;
    }

    {
        bool ok = whisper_sched_graph_init(vctx->sched, vctx->backends, [&]() {
            return whisper_vad_build_graph(*vctx);
        });

        if (!ok) {
            return false;
        }
    }

    return true;
}

struct whisper_vad_context *
whisper_vad_init_with_params(struct whisper_model_loader *loader,
                             struct whisper_vad_context_params params) {
    // Read the VAD model
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC) {
            return nullptr;
        }
    }

    whisper_vad_context *vctx = new whisper_vad_context;
    vctx->n_threads = params.n_threads;

    auto &model = vctx->model;
    auto &hparams = model.hparams;

    // load model context params.
    {
        int32_t str_len;
        read_safe(loader, str_len);
        std::vector<char> buffer(str_len + 1, 0);
        loader->read(loader->context, buffer.data(), str_len);
        std::string model_type(buffer.data(), str_len);
        model.type = model_type;

        int32_t major, minor, patch;
        read_safe(loader, major);
        read_safe(loader, minor);
        read_safe(loader, patch);
        std::string version_str = std::to_string(major) + "." +
                                  std::to_string(minor) + "." +
                                  std::to_string(patch);
        model.version = version_str;

        read_safe(loader, vctx->n_window);
        read_safe(loader, vctx->n_context);
    }

    // load model hyper params (hparams).
    {
        read_safe(loader, hparams.n_encoder_layers);

        hparams.encoder_in_channels = new int32_t[hparams.n_encoder_layers];
        hparams.encoder_out_channels = new int32_t[hparams.n_encoder_layers];
        hparams.kernel_sizes = new int32_t[hparams.n_encoder_layers];

        for (int32_t i = 0; i < hparams.n_encoder_layers; i++) {
            read_safe(loader, hparams.encoder_in_channels[i]);
            read_safe(loader, hparams.encoder_out_channels[i]);
            read_safe(loader, hparams.kernel_sizes[i]);
        }

        read_safe(loader, hparams.lstm_input_size);
        read_safe(loader, hparams.lstm_hidden_size);
        read_safe(loader, hparams.final_conv_in);
        read_safe(loader, hparams.final_conv_out);
    }

    // 1 STFT tensor, 4*2 encoder tensors, 4 LSTM tensors, 2 final output
    // tensors
    const size_t n_tensors = hparams.n_encoder_layers * 2 + 4 + 2 + 1;

    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };

            ggml_context *ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;
            model.ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    auto create_tensor = [&](vad_tensor type,
                             ggml_tensor *meta) -> ggml_tensor * {
        // ggml_op op = VAD_TENSOR_OPS.at(type);
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        ggml_context *ctx = get_ctx(buft);
        ggml_tensor *tensor = ggml_dup_tensor(ctx, meta);
        model.tensors[VAD_TENSOR_NAMES.at(type)] = tensor;

        return tensor;
    };

    // create tensors
    {
        ggml_init_params params = {
            /*.mem_size   =*/n_tensors * ggml_tensor_overhead(),
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };

        ggml_context *ctx = ggml_init(params);
        const auto &hparams = model.hparams;

        // SFTF precomputed basis matrix
        model.stft_forward_basis =
            create_tensor(VAD_TENSOR_STFT_BASIS,
                          ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, 1, 258));

        model.encoder_0_weight = create_tensor(
            VAD_TENSOR_ENC_0_WEIGHT,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.kernel_sizes[0],
                               hparams.encoder_in_channels[0],
                               hparams.encoder_out_channels[0]));
        model.encoder_0_bias =
            create_tensor(VAD_TENSOR_ENC_0_BIAS,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                             hparams.encoder_out_channels[0]));

        model.encoder_1_weight = create_tensor(
            VAD_TENSOR_ENC_1_WEIGHT,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.kernel_sizes[1],
                               hparams.encoder_in_channels[1],
                               hparams.encoder_out_channels[1]));
        model.encoder_1_bias =
            create_tensor(VAD_TENSOR_ENC_1_BIAS,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                             hparams.encoder_out_channels[1]));

        model.encoder_2_weight = create_tensor(
            VAD_TENSOR_ENC_2_WEIGHT,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.kernel_sizes[2],
                               hparams.encoder_in_channels[2],
                               hparams.encoder_out_channels[2]));
        model.encoder_2_bias =
            create_tensor(VAD_TENSOR_ENC_2_BIAS,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                             hparams.encoder_out_channels[2]));

        model.encoder_3_weight = create_tensor(
            VAD_TENSOR_ENC_3_WEIGHT,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hparams.kernel_sizes[3],
                               hparams.encoder_in_channels[3],
                               hparams.encoder_out_channels[3]));
        model.encoder_3_bias =
            create_tensor(VAD_TENSOR_ENC_3_BIAS,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                             hparams.encoder_out_channels[3]));

        // Hidden State dimension (input gate, forget gate, cell gate, output
        // gate)
        const int hstate_dim = hparams.lstm_hidden_size * 4;

        // LSTM weights - input to hidden
        model.lstm_ih_weight = create_tensor(
            VAD_TENSOR_LSTM_WEIGHT_IH,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.lstm_hidden_size,
                               hstate_dim));
        model.lstm_ih_bias =
            create_tensor(VAD_TENSOR_LSTM_BIAS_IH,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hstate_dim));

        // LSTM weights - hidden to hidden
        model.lstm_hh_weight = create_tensor(
            VAD_TENSOR_LSTM_WEIGHT_HH,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hparams.lstm_hidden_size,
                               hstate_dim));
        model.lstm_hh_bias =
            create_tensor(VAD_TENSOR_LSTM_BIAS_HH,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hstate_dim));

        // Final conv layer weight
        model.final_conv_weight = create_tensor(
            VAD_TENSOR_FINAL_CONV_WEIGHT,
            ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hparams.final_conv_in, 1));
        model.final_conv_bias =
            create_tensor(VAD_TENSOR_FINAL_CONV_BIAS,
                          ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1));

        ggml_free(ctx);
    }

    // allocate tensors in the backend buffers
    for (auto &p : ctx_map) {
        ggml_backend_buffer_type_t buft = p.first;
        ggml_context *ctx = p.second;
        ggml_backend_buffer_t buf =
            ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buf) {
            model.buffers.emplace_back(buf);
        }
    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded = 0;
        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            read_safe(loader, n_dims);
            read_safe(loader, length);
            read_safe(loader, ttype);

            if (loader->eof(loader->context)) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[4] = {1, 1, 1, 1};
            for (int i = 0; i < n_dims; ++i) {
                read_safe(loader, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> tmp(length);
            loader->read(loader->context, &tmp[0], tmp.size());
            name.assign(&tmp[0], tmp.size());

            if (model.tensors.find(name) == model.tensors.end()) {
                return nullptr;
            }

            auto tensor = model.tensors[name.data()];

            if (ggml_nelements(tensor) != nelements) {
                return nullptr;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] ||
                tensor->ne[2] != ne[2]) {
                return nullptr;
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) !=
                ggml_nbytes(tensor)) {
                return nullptr;
            }

            if (ggml_backend_buffer_is_host(tensor->buffer)) {
                // for the CPU and Metal backend, we can read directly into the
                // // tensor
                loader->read(loader->context, tensor->data,
                             ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device
                // memory
                read_buf.resize(ggml_nbytes(tensor));

                loader->read(loader->context, read_buf.data(), read_buf.size());

                ggml_backend_tensor_set(tensor, read_buf.data(), 0,
                                        ggml_nbytes(tensor));
            }

            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        if (model.n_loaded != (int)model.tensors.size()) {
            return nullptr;
        }
    }

    if (!whisper_vad_init_context(vctx)) {
        whisper_vad_free(vctx);
        return nullptr;
    }

    return vctx;
}

struct whisper_vad_context *whisper_vad_init_from_file_with_params(
    const char *path_model, struct whisper_vad_context_params params) {
#ifdef _MSC_VER
    std::wstring path_model_wide = utf8_to_wstring(path_model);
    auto fin = std::ifstream(path_model_wide, std::ios::binary);
#else
    auto fin = std::ifstream(path_model, std::ios::binary);
#endif
    if (!fin) {
        return nullptr;
    }

    whisper_model_loader loader = {};
    loader.context = &fin;

    loader.read = [](void *ctx, void *output, size_t read_size) {
        std::ifstream *fin = (std::ifstream *)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        return fin->eof();
    };

    loader.close = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        fin->close();
    };

    auto ctx = whisper_vad_init_with_params(&loader, params);
    if (!ctx) {
        whisper_vad_free(ctx);
        return nullptr;
    }
    ctx->path_model = path_model;
    return ctx;
}

static bool ggml_graph_compute_helper(ggml_backend_sched_t sched,
                                      struct ggml_cgraph *graph, int n_threads,
                                      bool sched_reset = true) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg =
            dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

        auto *fn_set_n_threads =
            (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_n_threads");
        if (fn_set_n_threads) {
            fn_set_n_threads(backend, n_threads);
        }
    }

    const bool t =
        (ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);

    if (!t || sched_reset) {
        ggml_backend_sched_reset(sched);
    }

    return t;
}

void whisper_vad_reset_state(struct whisper_vad_context *vctx) {
    // reset LSTM/GRU (and any other) state tensors
    ggml_backend_buffer_clear(vctx->buffer, 0);
    vctx->t_vad_us = 0;
}

bool whisper_vad_process_chunk(struct whisper_vad_context *vctx,
                               const float *samples, int n_samples,
                               float *out_prob) {
    // expects n_samples == vctx->n_window (or you do padding/accumulation
    // before)
    if (n_samples != vctx->n_window) {
        // you can pad or return false
        // for now, simple zero-padding:
        std::vector<float> window(vctx->n_window, 0.0f);
        const int to_copy = std::min(n_samples, vctx->n_window);
        std::copy(samples, samples + to_copy, window.begin());
        samples = window.data();
        n_samples = vctx->n_window;
    }

    auto &sched = vctx->sched.sched;

    // Lazily build graph once and cache in vctx
    if (!vctx->gf) {
        vctx->gf = whisper_vad_build_graph(*vctx);
        if (!ggml_backend_sched_alloc_graph(sched, vctx->gf)) {
            return false;
        }

        whisper_vad_reset_state(vctx);
    }

    ggml_cgraph *gf = vctx->gf;

    struct ggml_tensor *frame = ggml_graph_get_tensor(gf, "frame");
    struct ggml_tensor *prob = ggml_graph_get_tensor(gf, "prob");

    // Set input frame
    ggml_backend_tensor_set(frame, samples, 0,
                            ggml_nelements(frame) * sizeof(float));

    const int64_t t_start = ggml_time_us();

    // reuse scheduler, do not clear backend buffer → preserves RNN state
    if (!ggml_graph_compute_helper(sched, gf, vctx->n_threads, false)) {
        return false;
    }

    ggml_backend_tensor_get(prob, out_prob, 0, sizeof(float));

    vctx->t_vad_us += ggml_time_us() - t_start;

    // you can reset scheduler’s internal state if needed:
    ggml_backend_sched_reset(sched);

    return true;
}
