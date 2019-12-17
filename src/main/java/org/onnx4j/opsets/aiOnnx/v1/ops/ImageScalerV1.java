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
package org.onnx4j.opsets.aiOnnx.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.FloatsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * ImageScaler-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#ImageScaler-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
@Deprecated
public interface ImageScalerV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ImageScaler";

	public static final String ATTR_SCALE = "scale";

	public static final String ATTR_BIAS = "bias";

	/**
	 * 
	 * @param data
	 * @param shape
	 * @param consumedInputs
	 * @return
	 */
	public abstract T_TENSOR scale(T_TENSOR input, Float scale, List<Float> bias);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		Float scale = attrs.getAttrValue(ATTR_SCALE, FloatAttribute.class, 1.0F);

		List<Float> bias = attrs.getAttrValue(ATTR_BIAS, FloatsAttribute.class, null);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node, this.scale(inputArray[0].getTensor(), scale, bias));
	}

}