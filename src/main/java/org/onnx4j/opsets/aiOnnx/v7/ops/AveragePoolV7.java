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
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.aiOnnx.v1.ops.AveragePoolV1;

/**
 * AveragePool Operator v7
 * 
 * <p>
 * AveragePool consumes an input tensor X and applies average pooling across the
 * tensor according to kernel sizes, stride sizes, and pad lengths. average
 * pooling consisting of computing the average on all values of a subset of the
 * input tensor according to the kernel size and downsampling the data into the
 * output tensor Y for further processing
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 7
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#AveragePool-7">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool">
 *      ONNX.Operators.md</a>
 */
public interface AveragePoolV7<T_TENSOR> extends AveragePoolV1<T_TENSOR> {

	public static final String ATTR_COUNT_INCLUDE_PAD = "count_include_pad";

	/**
	 * Executes the operator
	 * 
	 * @param data
	 *            Input data tensor from the previous operator; dimensions for
	 *            image case are (N x C x H x W), where N is the batch size, C
	 *            is the number of channels, and H and W are the height and the
	 *            width of the data. For non image case, the dimensions are in
	 *            the form of (N x C x D1 x D2 ... Dn), where N is the batch
	 *            size. Optionally, if dimension denotation is in effect, the
	 *            operation expects the input data tensor to arrive with the
	 *            dimension denotation of [DATA_BATCH, DATA_CHANNEL,
	 *            DATA_FEATURE, DATA_FEATURE ...].
	 * @param autoPad
	 *            auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or
	 *            VALID. Where default value is NOTSET, which means explicit
	 *            padding is used. SAME_UPPER or SAME_LOWER mean pad the input
	 *            so that the output spatial size match the input.In case of odd
	 *            number add the extra padding at the end for SAME_UPPER and at
	 *            the beginning for SAME_LOWER. VALID mean no padding.
	 * @param kernelShape
	 *            The size of the kernel along each axis.
	 * @param pads
	 *            Padding for the beginning and ending along each spatial axis,
	 *            it can take any value greater than or equal to 0. The value
	 *            represent the number of pixels added to the beginning and end
	 *            part of the corresponding axis. `pads` format should be as
	 *            follow [x1_begin, x2_begin...x1_end, x2_end,...], where
	 *            xi_begin the number of pixels added at the beginning of axis
	 *            `i` and xi_end, the number of pixels added at the end of axis
	 *            `i`. This attribute cannot be used simultaneously with
	 *            auto_pad attribute. If not present, the padding defaults to 0
	 *            along start and end of each spatial axis.
	 * @param strides
	 *            Stride along each spatial axis.
	 * @param countIncludePad
	 *            Whether include pad pixels when calculating values for the
	 *            edges. Default is 0, doesn't count include pad.
	 * @return Output data tensor from average or max pooling across the input
	 *         tensor. Dimensions will vary based on various kernel, stride, and
	 *         pad sizes. Floor value of the dimension is used
	 */
	public abstract T_TENSOR averagePool(T_TENSOR data, String autoPad, List<Long> kernelShape, List<Long> pads,
			List<Long> strides, Long countIncludePad);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		Attributes attrs = node.getAttrs();

		//
		// string (default is NOTSET)
		//
		String autoPad = attrs.getAttrValue(ATTR_AUTO_PAD, StringAttribute.class, "NOTSET");

		//
		// list of ints (required)
		//
		List<Long> kernelShape = attrs.getAttrValue(ATTR_KERNEL_SHAPE, IntsAttribute.class, null);

		//
		// list of ints
		//
		List<Long> pads = attrs.getAttrValue(ATTR_PADS, IntsAttribute.class, null);

		//
		// list of ints
		//
		// Stride along each spatial axis.
		//
		List<Long> strides = attrs.getAttrValue(ATTR_STRIDES, IntsAttribute.class, null);

		//
		// int (default is 0)
		//
		Long countIncludePad = attrs.getAttrValue(ATTR_COUNT_INCLUDE_PAD, IntAttribute.class, 0L);

		Input[] inputArray = inputs.get();
		return Outputs.wrap(node,
				this.averagePool(inputArray[0].getTensor(), autoPad, kernelShape, pads, strides, countIncludePad));
	}

}