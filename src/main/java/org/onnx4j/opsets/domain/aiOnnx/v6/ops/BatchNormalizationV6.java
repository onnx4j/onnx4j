/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.domain.aiOnnx.v6.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.AiOnnxOperatorV6;
import org.onnx4j.opsets.operator.OperatorOutputs;

/**
 * BatchNormalization Operator v6
 * 
 * <p>
 * Carries out batch normalization as described in the paper
 * https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
 * there are multiple cases for the number of outputs, which we list below:
 * 
 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode) Output
 * case #2: Y (test mode)
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-6">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization">
 *      ONNX.Operators.md</a>
 */
public interface BatchNormalizationV6 extends BatchNormalizationV1, AiOnnxOperatorV6  {

	/**
	 * Executes operator
	 * 
	 * 
	 * 
	 * @param x
	 *            The input 4-dimensional tensor of shape NCHW.
	 * @param scale
	 *            The scale as a 1-dimensional tensor of size C to be applied to
	 *            the output.
	 * @param b
	 *            The bias as a 1-dimensional tensor of size C to be applied to
	 *            the output.
	 * @param mean
	 *            The running mean (training) or the estimated mean (testing) as
	 *            a 1-dimensional tensor of size C.
	 * @param var
	 *            The running variance (training) or the estimated variance
	 *            (testing) as a 1-dimensional tensor of size C.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @param epsilon
	 *            The epsilon value to use to avoid division by zero, default is
	 *            1e-5f.
	 * @param momentum
	 *            Factor used in computing the running mean and variance.e.g.,
	 *            running_mean = running_mean * momentum + mean * (1 -
	 *            momentum), default is 0.9f.
	 * @param spatial
	 *            If true, compute the mean and variance across all spatial
	 *            elements If false, compute the mean and variance across per
	 *            feature.Default is 1.
	 * @return
	 * 
	 * 		<pre>
	 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
	 *         </pre>
	 * 
	 *         <pre>
	 * Output case #2: Y (test mode)
	 *         </pre>
	 */
	/*public abstract T_TENSOR[] batchNormalization(T_TENSOR x, T_TENSOR scale, T_TENSOR b, T_TENSOR mean, T_TENSOR var,
			Float epsilon, Float momentum, Boolean spatial, Boolean isTest);*/

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class BatchNormalizationInputsV6<T_TENSOR> extends BatchNormalizationInputsV1<T_TENSOR> {

		/**
		 * Executes operator
		 * 
		 * @param x
		 *            The input 4-dimensional tensor of shape NCHW.
		 * @param scale
		 *            The scale as a 1-dimensional tensor of size C to be
		 *            applied to the output.
		 * @param b
		 *            The bias as a 1-dimensional tensor of size C to be applied
		 *            to the output.
		 * @param mean
		 *            The running mean (training) or the estimated mean
		 *            (testing) as a 1-dimensional tensor of size C.
		 * @param var
		 *            The running variance (training) or the estimated variance
		 *            (testing) as a 1-dimensional tensor of size C.
		 * @param epsilon
		 *            The epsilon value to use to avoid division by zero,
		 *            default is 1e-5f.
		 * @param isTest
		 *            If set to nonzero, run spatial batch normalization in test
		 *            mode, default is 0.
		 * @param momentum
		 *            Factor used in computing the running mean and
		 *            variance.e.g., running_mean = running_mean * momentum +
		 *            mean * (1 - momentum), default is 0.9f.
		 * @param spatial
		 *            If true, compute the mean and variance across all spatial
		 *            elements If false, compute the mean and variance across
		 *            per feature.Default is 1.
		 * @return
		 * 
		 * 		<pre>
		 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
		 *         </pre>
		 * 
		 *         <pre>
		 * Output case #2: Y (test mode)
		 *         </pre>
		 */
		public BatchNormalizationInputsV6(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Override
		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException(
					String.format("Attribute named \"%s\" has deprecated", ATTR_CONSUMED_INPUTS));
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class BatchNormalizationOutputsV6<T_TENSOR> extends BatchNormalizationOutputsV1<T_TENSOR> {

		public BatchNormalizationOutputsV6(OperatorOutputs<T_TENSOR> operatorOutputs) {
			super(operatorOutputs);
		}

		public BatchNormalizationOutputsV6(T_TENSOR y) {
			this(y, null, null, null, null);
		}

		public BatchNormalizationOutputsV6(T_TENSOR data, T_TENSOR mean, T_TENSOR var, T_TENSOR savedMean,
				T_TENSOR savedVar) {
			super(data, mean, var, savedMean, savedVar);
		}

	}

}