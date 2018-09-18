if ~exist("trump2016.mat", "file")

    fname = 'tweets/condensed_2016.json'; 
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    tweet = jsondecode(str);
    tweet_number = length(tweet);

    u = tweet(1).text;
    for i=2:tweet_number
        u = join([u, tweet(i).text]);
    end
    end_of_tweet = 'ŋ';
    tweet_chars = join([unique(u), end_of_tweet]);
    K = length(tweet_chars);

    char_to_ind = containers.Map(num2cell(tweet_chars), 1:K);
    ind_to_char = containers.Map(1:K, num2cell(tweet_chars));

    for i=1:tweet_number
        len = length(tweet(i).text);
        X_chars{i} = zeros(K, len);
        Y_chars{i} = zeros(K, len);

        for j=1:len-1
            X_chars{i}(char_to_ind(tweet(i).text(j)), j) = 1;
            Y_chars{i}(char_to_ind(tweet(i).text(j+1)), j) = 1;
        end
        X_chars{i}(char_to_ind(tweet(i).text(len)), len) = 1;
        Y_chars{i}(K, len) = 1; % end of tweet
    end

    save("trump2016.mat", 'X_chars', 'Y_chars', 'K', 'char_to_ind', 'ind_to_char', 'tweet_number', 'end_of_tweet');
else
    load("trump2016.mat", 'X_chars', 'Y_chars', 'K', 'char_to_ind', 'ind_to_char', 'tweet_number', 'end_of_tweet');
end

% hidden state dimension
m = 100;

if exist("trump2016_param.mat", "file")
    load('trump2016_param.mat', 'LSTM')
    
    seq = syntethize_sequence(LSTM, zeros(m, 1), zeros(m, 1), X_chars{round(rand(1)*(K-1))}(:, 1), K);
    print_sequence(seq, K, ind_to_char);
else
    %--------------------------------------------------------------------------
    % hyperparameters & network initialization
    %--------------------------------------------------------------------------
    GDparam.epoch = 30;
    GDparam.eta = .1;
    GDparam.epsilon = 1e-8;
    sig = .01;

    LSTM.b = zeros(4*m, 1);
    LSTM.d = zeros(K, 1);
    LSTM.Uall = randn(4*m, K)*sig;
    LSTM.Wall = randn(4*m, m)*sig;
    LSTM.V = randn(K, m)*sig;

    LSTM = AdaGrad(GDparam, LSTM, X_chars, Y_chars, m, ind_to_char, K);
end
%--------------------------------------------------------------------------
% functions
%--------------------------------------------------------------------------

function P = softmax(s)
    P = exp(s);
    for i=1:size(s, 2)
        P(:, i) = P(:, i) ./ sum(P(:, i));
    end
end

function sigma = sigmoid(x)
    sigma = 1./(1 + exp(-x));
end

function Y = syntethize_sequence(LSTM, c0, h0, x0, K)
    m = size(LSTM.b, 1) / 4;
    tweet_max_length = 280;
    
    Y = zeros(length(LSTM.d), 1);
    Y(1) = find(x0 == 1);
    
    E = eye(4*m);

    c = c0;
    h = h0;
    x = x0;
    for n=2:tweet_max_length
        a = LSTM.Wall * h + LSTM.Uall * x + LSTM.b;
        
        f = sigmoid(a(1:m));
        i = sigmoid(a(m+1:2*m));
        ctilde = tanh(a(2*m+1:3*m));
        out = sigmoid(a(3*m+1:4*m));
        
        c = f .* c + i .* ctilde;
        h = out .* tanh(c);
        
        o = LSTM.V * h + LSTM.d;
        p = softmax(o);
        
        % sample character
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a > 0);
        ii = ixs(1);
        
        if ii == K
            break;
        end
        
        Y(n) = ii;
        
        x = zeros(size(x));
        x(ii) = 1;
    end
end

function print_sequence(seq, K, ind_to_char)
    out = '';
    for i=1:length(seq)
        if seq(i) ~= K && isKey(ind_to_char, seq(i))
            out = [out, ind_to_char(seq(i))];
        end
    end
    fprintf('%s\n', out)
end

function L = compute_loss(P, Y)
    N = size(P,2);

    L = 0;
    for i=1:N
        L = L - log(Y(:, i)' * P(:, i));
    end
end

function l = ComputeLoss(X, Y, LSTM, hprev)
    P = forward(LSTM, X, hprev);

    l = compute_loss(P, Y);
end

function [P, A, H, C, gates] = forward(LSTM, X_chars, c, h)
    T = size(X_chars, 2);
    m = size(LSTM.b, 1) / 4;
    
    P = zeros(size(LSTM.d, 1), T);
    H = zeros(m, T+1);
    C = zeros(m, T+1);
    A = zeros(4*m, T);
    E = eye(4*m);
    
    gates.F = zeros(m, T+1);
    gates.I = zeros(m, T);
    gates.Ctilde = zeros(m, T);
    gates.O = zeros(m, T);
    
    C(:, 1) = c;
    H(:, 1) = h;
    for t=1:T
        A(:, t) = LSTM.Wall * H(:, t) + LSTM.Uall * X_chars(:, t) + LSTM.b;
        
        gates.F(:, t) = sigmoid(E(1:m, :) * A(:, t));
        gates.I(:, t) = sigmoid(E(m+1:2*m, :) * A(:, t));
        gates.Ctilde(:, t) = tanh(E(2*m+1:3*m, :) * A(:, t));
        gates.O(:, t) = sigmoid(E(3*m+1:4*m, :) * A(:, t));
        
        C(:, t+1) = gates.F(:, t) .* C(:, t) + gates.I(:, t) .* gates.Ctilde(:, t);
        H(:, t+1) = gates.O(:, t) .* tanh(C(:, t+1));
        
        o = LSTM.V * H(:, t+1) + LSTM.d;
        P(:, t) = softmax(o);
    end
    
end

function [grads, loss, c, h] = ComputeGradients(LSTM, X, Y, c, h)
    m = length(LSTM.b) / 4;

    [P, A, H, C, gates] = forward(LSTM, X, c, h);
    loss = compute_loss(P, Y) / size(X, 2);
    c = C(:, size(C,2));
    h = H(:, size(H,2));
    
    grads.V = zeros(size(LSTM.V));
    grads.Wall = zeros(size(LSTM.Wall));
    grads.Uall = zeros(size(LSTM.Uall));
    grads.b = zeros(size(LSTM.b));
    grads.d = zeros(size(LSTM.d));
    
    grad_a = zeros(size(LSTM.Wall, 1), 1);
    grad_c = zeros(size(C, 1), 1);
    for t=size(X, 2):-1:1
        g = -(Y(:, t) - P(:, t))';
        grads.V = grads.V + g' * H(:, t+1)'; 
        grads.d = grads.d + g';
        
        tanh_C = tanh(C(:, t+1));
        
        grad_h = g * LSTM.V + grad_a' * LSTM.Wall; 
        grad_c = grad_h' .* gates.O(:, t) .* (1 - tanh_C.^2) + grad_c .* gates.F(:, t+1); % CHECK t+1
        
        grad_f = grad_c .* C(:, t) .* sigmoid(A(1:m, t)) .* (1 - sigmoid(A(1:m, t)));
        grad_i = grad_c .* gates.Ctilde(:, t) .* sigmoid(A(m+1:2*m, t)) .* (1 - sigmoid(A(m+1:2*m, t)));
        grad_ctilde = grad_c .* gates.I(:, t) .* (1 - tanh(A(2*m+1:3*m, t)) .^ 2);
        grad_o = grad_h' .* tanh_C .* sigmoid(A(3*m+1:4*m, t)) .* (1 - sigmoid(A(3*m+1:4*m, t)));
        
        grad_a = [grad_f; grad_i; grad_ctilde; grad_o];
        
        grads.Wall = grads.Wall + grad_a * H(:, t)';
        grads.Uall = grads.Uall + grad_a * X(:, t)';
        grads.b = grads.b + grad_a;
    end    
end

function LSTM = AdaGrad(GDparam, LSTM, X, Y, m, ind_to_char, K)

    smooth_loss = zeros(GDparam.epoch * length(X), 1);
    sum_grads.V = zeros(size(LSTM.V));
    sum_grads.W = zeros(size(LSTM.Wall));
    sum_grads.U = zeros(size(LSTM.Uall));
    sum_grads.b = zeros(size(LSTM.b));
    sum_grads.d = zeros(size(LSTM.d));
    
    update = 0;
    boh = 0;
    for e=1:GDparam.epoch
        
        hprev = zeros(m, 1);
        cprev = zeros(m, 1);
        
        perm = randperm(length(X));
        
        for s=1:length(X)
            update = update + 1;
            
            [grads, train_loss, cprev, hprev] = ComputeGradients(LSTM, X{perm(s)}, Y{perm(s)}, cprev, hprev);

%             % gradient clipping
%             for f = fieldnames(grads)'
%                 grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
%             end
            
            % AdaGrad
            sum_grads.V = sum_grads.V + grads.V .^2;
            sum_grads.W = sum_grads.W + grads.Wall .^2;
            sum_grads.U = sum_grads.U + grads.Uall .^2;
            sum_grads.b = sum_grads.b + grads.b .^2;
            sum_grads.d = sum_grads.d + grads.d .^2;
            
            LSTM.V = LSTM.V - GDparam.eta ./ sqrt(sum_grads.V + GDparam.epsilon) .* grads.V;
            LSTM.Wall = LSTM.Wall - GDparam.eta ./ sqrt(sum_grads.W + GDparam.epsilon) .* grads.Wall;
            LSTM.Uall = LSTM.Uall - GDparam.eta ./ sqrt(sum_grads.U + GDparam.epsilon) .* grads.Uall;
            LSTM.b = LSTM.b - GDparam.eta ./ sqrt(sum_grads.b + GDparam.epsilon) .* grads.b;
            LSTM.d = LSTM.d - GDparam.eta ./ sqrt(sum_grads.d + GDparam.epsilon) .* grads.d;

            % update loss
            if update == 1
                smooth_loss(update) = train_loss;
            else
                smooth_loss(update) = 0.999 * smooth_loss(update-1) + 0.001 * train_loss;
            end
            
            % print synthetized sequence
            if mod(update, 500) == 0 || update == 1
                fprintf('epoch: %i, iteration: %i, smoothed loss: %i\n', e, update, smooth_loss(update))
                seq = syntethize_sequence(LSTM, zeros(m, 1), zeros(m, 1), X{perm(s)}(:, 1), K);
                print_sequence(seq, K, ind_to_char);
            end
        end
    end

    figure, 
    plot(1:update, smooth_loss(1:update))
    
    % save model
    save("trump2016_param.mat", 'LSTM');
end
