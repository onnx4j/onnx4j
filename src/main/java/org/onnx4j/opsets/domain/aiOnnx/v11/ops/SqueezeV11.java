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
package org.onnx4j.opsets.domain.aiOnnx.v11.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.SqueezeV1;
import org.onnx4j.opsets.domain.aiOnnx.v11.AiOnnxOperatorV11;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;

import com.google.common.collect.Lists;

/**
 * Squeeze Operator v11
 * 
 * <p>
 * Remove single-dimensional entries from the shape of a tensor. Takes a
 * parameter axes with a list of axes to squeeze. If axes is not provided, all
 * the single dimensions will be removed from the shape. If an axis is selected
 * with shape entry not equal to one, an error is raised.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 11
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Squeeze-11">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze">
 *      ONNX.Operators.md</a>
 */
public interface SqueezeV11 extends SqueezeV1, AiOnnxOperatorV11 {

	class SqueezeInputsV11<T_TENSOR> extends SqueezeInputsV1<T_TENSOR> {

		public SqueezeInputsV11(Node node, Inputs inputs) {
			super(node, inputs);

			dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// int (default is 1)
			//
			axesField = new AttributeField<List<Long>>(super.attrs, ATTR_AXES, IntsAttribute.class,
					Lists.newLinkedList(), true);
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SqueezeOutputV11<T_TENSOR> extends SqueezeOutputV1<T_TENSOR> {

		public SqueezeOutputV11(T_TENSOR output) {
			super(output);
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Tensors with at least max(dims) dimensions.
	 * @param axes
	 *            List of integers indicating the dimensions to squeeze.
	 *            Negative value means counting dimensions from the back.
	 *            Accepted range is [-r, r-1] where r = rank(data).
	 * 
	 * @return Reshaped tensor with same data as input.
	 */
	// public abstract T_TENSOR squeeze(T_TENSOR data, List<Long> axes);

}