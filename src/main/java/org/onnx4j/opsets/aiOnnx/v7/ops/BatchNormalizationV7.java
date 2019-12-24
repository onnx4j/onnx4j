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
package org.onnx4j.opsets.aiOnnx.v7.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.ops.BatchNormalizationV1;

/**
 * BatchNormalization Operator v7
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
 * @version 7
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-7">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization">
 *      ONNX.Operators.md</a>
 */
public interface BatchNormalizationV7<T_TENSOR> extends BatchNormalizationV1<T_TENSOR> {

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
	 *         <pre>
	 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
	 *         </pre>
	 * 
	 *         <pre>
	 * Output case #2: Y (test mode)
	 *         </pre>
	 */
	public abstract T_TENSOR[] batchNormalization(T_TENSOR x, T_TENSOR scale, T_TENSOR b, T_TENSOR mean, T_TENSOR var,
			List<Long> consumedInputs, Float epsilon, Float momentum, Boolean spatial);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// Legacy optimization attribute.
		//
		List<Long> consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

		//
		// The epsilon value to use to avoid division by zero, default is 1e-5f.
		//
		Float epsilon = attrs.getAttrValue(ATTR_EPSILON, FloatAttribute.class, 1e-5f);

		//
		// Factor used in computing the running mean and variance.e.g.,
		// running_mean = running_mean * momentum + mean * (1 - momentum),
		// default is 0.9f.
		//
		Float momentum = attrs.getAttrValue(ATTR_MOMENTUM, FloatAttribute.class, 0.9f);

		//
		// If true, compute the mean and variance across all spatial elements If
		// false, compute the mean and variance across per feature.Default is 1.
		//
		Boolean spatial = attrs.getAttrValue(ATTR_SPATIAL, IntAttribute.class, 1L).intValue() == 1 ? true : false;

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.batchNormalization(inputArray[0].getTensor(), inputArray[1].getTensor(), inputArray[2].getTensor(),
						inputArray[3].getTensor(), inputArray[4].getTensor(), consumedInputs, epsilon, momentum,
						spatial));
	}

}